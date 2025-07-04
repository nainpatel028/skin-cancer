import torch
import pandas as pd
import random
import numpy as np
import os
from sklearn.model_selection import train_test_split as tts
import seaborn as sns
from torchvision import transforms, models  # Added models import
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from data_loader import FlowerDataset  # Import from the new data_loader file
from train import train, plot, preds,device  # Importing functions from train.py
import torch.nn as nn
import torch.optim as optim


# Seed function for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()

# Load the metadata and update the path list
train_metadata_selected_features = pd.read_csv('train-metadata.csv')  # Replace with actual CSV path
path_list = [f"train-image/image/{id}.jpg" for id in train_metadata_selected_features['isic_id']]
train_metadata_selected_features["path_list"] = pd.Series(path_list)

# Balance the dataset (3:1 nonmelanoma to melanoma)
melanoma = train_metadata_selected_features[train_metadata_selected_features["target"] == 1]
nonmelanoma = train_metadata_selected_features[train_metadata_selected_features["target"] == 0].sample(len(melanoma) * 3)
Data = pd.concat([melanoma, nonmelanoma], axis=0).reset_index(drop=True)
train_metadata_selected_features = Data[["path_list", "target"]]

# Visualize the distribution of target labels
sns.countplot(x=Data["target"], palette="cool")

# Define transformations for training and testing
transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Encode target labels
label_encoder = LabelEncoder()
train_metadata_selected_features.iloc[:, 1] = label_encoder.fit_transform(train_metadata_selected_features.iloc[:, 1])

# Display class labels
for i, class_name in enumerate(label_encoder.classes_):
    print(f"class: {class_name}, Label: {i}")

# Split the dataset into Train, Validation, and Test sets
Train, Test = tts(train_metadata_selected_features, test_size=0.01/2, stratify=train_metadata_selected_features.iloc[:, 1])
Train, Valid = tts(Train, test_size=0.1, stratify=Train.iloc[:, 1])

# Print dataset shapes
print(f"Train Shape is: {Train.shape}")
print(f"Valid Shape is: {Valid.shape}")
print(f"Test Shape is: {Test.shape}")
print(f"Validation and Test Len is {(Valid.shape[0] + Test.shape[0]) / train_metadata_selected_features.shape[0]:.2%}")

# Create dataset objects
train_ds = FlowerDataset(Train, transform)
valid_ds = FlowerDataset(Valid, transform_test)
test_ds = FlowerDataset(Test, transform_test)

# Create data loaders
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=32, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

# Model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.inception_v3(pretrained=True)
model.aux_logits = False
model.fc = nn.Linear(model.fc.in_features, len(label_encoder.classes_))

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training
train(model, optimizer, criterion, train_dl, valid_dl, epochs=10, save_dir="models")

# Plot results
plot()

# Testing predictions
preds(model, test_dl, label_encoder, device)
