Skin Cancer Detection with 3D-TBP

Description
This project is designed for the ISIC 2024 Challenge, focusing on skin lesion classification using machine learning and deep learning techniques. The repository includes scripts for data preprocessing, training models, and deploying an application for inference.

Installation

1. Set Up the Environment
Create a new Python environment named skindet:
--python -m venv skindet  
--source skindet/bin/activate   # On Windows, use: skindet\Scripts\activate  

2. Install Dependencies
Install the required libraries using the requirements.txt file:

--pip install -r requirements.txt  

Dataset Preparation

Download Dataset

Go to the ISIC 2024 Challenge Dataset.
Download the train-images and train-metadata.csv dataset.
https://www.kaggle.com/competitions/isic-2024-challenge/data

Organize Dataset

Place the images into the following folder structure:


train-image/  
└── image/  
    ├── image_1.jpg  
    ├── image_2.jpg  
    └── ... 

Usage
1. Start Training
To begin training the model, run:
--python main.py  

2. Run the Application
To launch the app for inference:
--python app.py 

3. Run the Jupyter Notebook
To see detailed steps for EDA, preprocessing, data splitting, and image processing, open the .ipynb file using Jupyter Notebook or JupyterLab:
--jupyter notebook <final>.ipynb

There is plot name plot.png and ResNet50.ong in that you will find the training and validation for VGG19 and ResNet50 respectively.

Download the model from the drive to run the code as the limit is of 100 MB while submitting the file so we are uploading the model to drive the link of drive is below
https://drive.google.com/drive/u/1/folders/1dGZJy_S6OT1Hjczfi4Rcoiq-y-BXjG52

Follow the file structure for model
└── models/  
    ├── model_epoch_50.pth

File Structure

project-directory/  
│
├── train-image/          # Dataset folder  
│   └── image/            # Images folder  
├── scripts/              # Python scripts for training and evaluation  
├── app.py                # Application for inference  
├── main.py               # Script to start model training  
├── requirements.txt      # Dependency list  
└── README.md             # Project documentation  
└── train-metadata.csv    #csv file 
├── notebooks/          # notebook 
│   └── final.ipynb 
├── models/          # notebook 
│   └── model_epoch_50.pth  #VGG19 model
│   └── ResNet50_model_epoch_50.pth #ResNet50 model
├── data_loader.py
├── image_augmentation.py
├── preprocess.py
├── train.py

If you are getting an error like OSError: [Errno 48] Address already in use because we are using static address
then destroy the terminal and run it again
--python app.py
