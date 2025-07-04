# preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# Function for data preprocessing
def preprocess_data(input_file):
    df = pd.read_csv(input_file)
    
    # Fill missing values
    df['age_approx'].fillna(df['age_approx'].median(), inplace=True)
    df['sex'].fillna(df['sex'].mode()[0], inplace=True)
    df['anatom_site_general'].fillna(df['anatom_site_general'].mode()[0], inplace=True)
    
    # Encoding categorical variables
    train_metadata = pd.get_dummies(df, columns=['sex', 'anatom_site_general'], drop_first=True)
    
    # Continuous columns for normalization
    continuous_columns = [
        'age_approx', 'clin_size_long_diam_mm', 'mel_thick_mm',
        'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext',
        'tbp_lv_C', 'tbp_lv_Cext', 'tbp_lv_H', 'tbp_lv_Hext',
        'tbp_lv_L', 'tbp_lv_Lext', 'tbp_lv_areaMM2',
        'tbp_lv_perimeterMM', 'tbp_lv_color_std_mean',
        'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL',
        'tbp_lv_deltaLB', 'tbp_lv_deltaLBnorm', 'tbp_lv_eccentricity',
        'tbp_lv_minorAxisMM', 'tbp_lv_nevi_confidence',
        'tbp_lv_norm_border', 'tbp_lv_norm_color',
        'tbp_lv_radial_color_std_max', 'tbp_lv_stdL',
        'tbp_lv_stdLExt', 'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle',
        'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z', 'tbp_lv_dnn_lesion_confidence'
    ]
    scaler = StandardScaler()
    train_metadata[continuous_columns] = scaler.fit_transform(train_metadata[continuous_columns])
    
    # Drop unnecessary columns
    columns_to_drop = ['image_type', 'tbp_lv_location_simple', 'copyright_license',
                       'lesion_id', 'iddx_full', 'iddx_1', 'iddx_2', 'iddx_3',
                       'iddx_4', 'iddx_5', 'mel_mitotic_index', 'mel_thick_mm',
                       'patient_id']
    train_metadata.drop(columns=columns_to_drop, inplace=True)
    
    # Create interaction features
    train_metadata['aspect_ratio'] = train_metadata['clin_size_long_diam_mm'] / train_metadata['tbp_lv_minorAxisMM']
    train_metadata['size_perimeter'] = train_metadata['clin_size_long_diam_mm'] * train_metadata['tbp_lv_perimeterMM']
    train_metadata['area_perimeter'] = train_metadata['tbp_lv_areaMM2'] * train_metadata['tbp_lv_perimeterMM']
    train_metadata['color_variability_deltaA'] = train_metadata['tbp_lv_color_std_mean'] * train_metadata['tbp_lv_deltaA']
    train_metadata['color_variability_deltaB'] = train_metadata['tbp_lv_color_std_mean'] * train_metadata['tbp_lv_deltaB']
    train_metadata['color_variability_deltaL'] = train_metadata['tbp_lv_color_std_mean'] * train_metadata['tbp_lv_deltaL']
    train_metadata['color_asymmetry_symmetry'] = train_metadata['tbp_lv_radial_color_std_max'] * train_metadata['tbp_lv_symm_2axis']
    train_metadata['symmetry_eccentricity'] = train_metadata['tbp_lv_symm_2axis'] * train_metadata['tbp_lv_eccentricity']
    train_metadata['size_x_coord'] = train_metadata['clin_size_long_diam_mm'] * train_metadata['tbp_lv_x']
    train_metadata['nevi_confidence_border_irregularity'] = train_metadata['tbp_lv_nevi_confidence'] * train_metadata['tbp_lv_norm_border']
    train_metadata['color_border_irregularity'] = train_metadata['tbp_lv_norm_color'] * train_metadata['tbp_lv_norm_border']
    train_metadata['age_nevi_confidence'] = train_metadata['age_approx'] * train_metadata['tbp_lv_nevi_confidence']
    train_metadata['age_symmetry'] = train_metadata['age_approx'] * train_metadata['tbp_lv_symm_2axis']
    
    # One-hot encoding for location column
    train_metadata = pd.get_dummies(train_metadata, columns=['tbp_lv_location'], drop_first=True)
    
    # Separate target and ID columns
    isic_id = train_metadata.pop('isic_id')
    target = train_metadata.pop('target')
    
    # Define preprocessing for numeric and boolean features
    numeric_features = train_metadata.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    boolean_features = train_metadata.select_dtypes(include=['bool']).columns.tolist()
    boolean_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('bool', boolean_transformer, boolean_features)
        ])
    
    # PCA pipeline
    pca_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=25))
    ])
    
    # Apply PCA
    pca_result = pca_pipeline.fit_transform(train_metadata)
    train_metadata_pca = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
    train_metadata_pca['isic_id'] = isic_id.values
    train_metadata_pca['target'] = target.values
    
    return train_metadata_pca

# Function for ML preprocessing (train-test split and SMOTE)
def ml_preprocess_data(train_metadata_selected_features):
    X = train_metadata_selected_features.drop(columns=['target'])
    y = train_metadata_selected_features['target']

    if 'isic_id' in X.columns:
        isic_id = X['isic_id']
        X = X.drop(columns=['isic_id'])
    else:
        isic_id = None

    from sklearn.model_selection import train_test_split as tts
    X_train, X_val, y_train, y_val = tts(X, y, test_size=0.2, random_state=42, stratify=y)

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    if isic_id is not None:
        new_isic_ids = ['synthetic_' + str(i) for i in range(len(X_train_smote) - len(X_train))]
        original_isic_ids = isic_id.iloc[X_train.index].tolist()
        X_train_smote = pd.DataFrame(X_train_smote, columns=X_train.columns)
        X_train_smote['isic_id'] = original_isic_ids + new_isic_ids
        X_val['isic_id'] = isic_id.iloc[X_val.index].values
    else:
        print("'isic_id' was not found in the original data.")

    num_negatives = sum(y_train_smote == 0)
    num_positives = sum(y_train_smote == 1)
    scale_pos_weight = num_negatives / num_positives
    print(f'Scale Pos Weight: {scale_pos_weight}')

    if 'isic_id' in X_train_smote.columns:
        X_train_smote = X_train_smote.drop(columns=['isic_id'])
    if 'isic_id' in X_val.columns:
        X_val = X_val.drop(columns=['isic_id'])

    return X_train_smote, X_val, y_train_smote, y_val, scale_pos_weight
