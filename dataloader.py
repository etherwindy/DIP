import os
import pandas as pd
import cv2

import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

from preprocess import preprocess_image

def random_rotation(image, max_angle=30):
    angle = np.random.uniform(-max_angle, max_angle)
    return scipy.ndimage.rotate(image, angle, reshape=False, mode='nearest')

class MyDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None):
        self.image_dir = image_dir
        self.csv_file = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.csv_file.iloc[idx, 0])

        image = cv2.imread(image_path)
        image = preprocess_image(image)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        image = image / 255.0
        image = random_rotation(image, max_angle=10)
        image = np.moveaxis(image, -1, 0)
        label = self.csv_file.iloc[idx, 1]

        return image, label


def create_dataloader(train_batch_size: int, val_batch_size: int):
    dataset = MyDataset("./dataset/image", "./dataset/label.csv")

    train_size = int(0.8 * len(dataset)) 
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    #train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=16, pin_memory=True)
    train_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=16, pin_memory=True)

    return train_loader, val_loader
