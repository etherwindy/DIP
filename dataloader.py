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

class MyDatasetWithPreprocess(Dataset):
    def __init__(self, image_dir, csv_file, img_size=(512, 512), transform=None):
        self.image_dir = image_dir
        self.csv_file = pd.read_csv(csv_file)
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.csv_file.iloc[idx, 0])

        image = cv2.imread(image_path)
        image = preprocess_image(image)
        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)
        image = image / 255.0
        image = random_rotation(image, max_angle=20)
        image = np.moveaxis(image, -1, 0)
        label = self.csv_file.iloc[idx, 1]

        return image, label

class MyDataset(Dataset):
    def __init__(self, image_dir, csv_file, img_size=(512, 512), transform=None):
        self.image_dir = image_dir
        self.csv_file = pd.read_csv(csv_file)
        self.transform = transform
        self.img_size = img_size
        self.sigma = 20
        self.radius = self.img_size[0] // 2
        X, Y = np.ogrid[:2*self.radius, :2*self.radius]
        self.mask = (X - self.radius)**2 + (Y - self.radius)**2 > self.radius**2

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.csv_file.iloc[idx, 0])

        image = cv2.imread(image_path)
        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)
        image = random_rotation(image, max_angle=180)
        gaussian = np.random.normal(0, self.sigma, image.shape).astype(np.int16)
        image = image.astype(np.int16)
        image = cv2.add(image, gaussian)
        image = image.clip(min=0, max=255).astype(np.uint8)
        image[self.mask] = 0
        image = image.astype(np.float32)
        image = np.moveaxis(image, -1, 0)
        label = self.csv_file.iloc[idx, 1]

        return image, label


def create_dataloader(train_batch_size: int, val_batch_size: int, img_size=(512, 512), preprocess=False):
    if preprocess:
        train_dataset = MyDatasetWithPreprocess("./dataset/image_train", "./dataset/label_train.csv", img_size=img_size)
        val_dataset = MyDatasetWithPreprocess("./dataset/image_test", "./dataset/label_test.csv", img_size=img_size)
    else:
        train_dataset = MyDataset("./dataset/image_train", "./dataset/label_train.csv", img_size=img_size)
        val_dataset = MyDataset("./dataset/image_test", "./dataset/label_test.csv", img_size=img_size)

    #train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=16, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=16, pin_memory=True)

    return train_loader, val_loader
