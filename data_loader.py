import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class FlowerDataset(Dataset):
    def __init__(self, train_metadata_selected_features, transform=None, test_mode=False):
        self.train_metadata_selected_features = train_metadata_selected_features.reset_index(drop=True)
        self.path = self.train_metadata_selected_features.iloc[:, 0]
        
        self.test_mode = test_mode
        if not self.test_mode:
            self.target = self.train_metadata_selected_features.iloc[:, 1]
        self.transform = transform
    
    def __getitem__(self, index):
        # Open the image file
        img = Image.open(self.path[index])
        
        # Apply transformations if specified
        if self.transform is not None:
            img = self.transform(img)
        
        target = self.target[index]
        
        if not self.test_mode:
            return img, target
        else:
            return img
    
    def __len__(self):
        return len(self.path)
