import os
import cv2

import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy
from torch.utils.data import Dataset, DataLoader

def preprocess_image(image, IsGray: bool = False, radius=256):
    if IsGray:  # return a gray preprocessed image
        image = np.uint8(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        gamma = 1.2
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        image = cv2.LUT(image, table)
    else:  # return a RGB preprocessed image (looks better)
        RADIUS = radius
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), RADIUS / 30), -4, 128)
        circle = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(circle, (RADIUS, RADIUS), int(RADIUS * 0.9), (1, 1, 1), -1, 8, 0)
        image = image * circle + 128 * (1 - circle)

    return image

def random_rotation(image, max_angle=30):
    angle = np.random.uniform(-max_angle, max_angle)
    return scipy.ndimage.rotate(image, angle, reshape=False, mode='nearest')

def random_flip(image):
    if np.random.rand() < 0.5:
        image = np.fliplr(image)
        image = image.copy()
    return image

def random_bright(image, delta=20, radius=256):
    if np.random.rand() < 0.5:
        X, Y = np.ogrid[:2*radius, :2*radius]
        mask = (X - radius)**2 + (Y - radius)**2 < (radius * 0.9)**2
        delta = np.random.randint(-delta, delta)
        image = image.astype(np.int16)
        image[mask] += delta
        image = image.clip(min=0, max=255).astype(np.uint8)
    return image

def random_noise(image, sigma=10, radius=256):
    if np.random.rand() < 0.5:
        gaussian = np.random.normal(0, sigma, image.shape).astype(np.int16)
        image = image.astype(np.int16)
        image = cv2.add(image, gaussian)
        image = image.clip(min=0, max=255).astype(np.uint8)
        X, Y = np.ogrid[:2*radius, :2*radius]
        mask = (X - radius)**2 + (Y - radius)**2 > (radius*0.9)**2
        image[mask] = 128
    return image

def random_mask(image, radius=256, percent=0.2):
    if np.random.rand() < 0.5:
        max_size = int(radius * percent)
        mask_up = np.random.randint(0, max_size)
        mask_down = np.random.randint(0, max_size)
        image[:mask_up, :, :] = 128
        image[-mask_down:, :, :] = 128
    return image

def random_augmentation(image, radius=256):
    image = random_rotation(image)
    image = random_flip(image)
    image = random_bright(image, radius=radius)
    image = random_noise(image, radius=radius)
    #image = random_mask(image, radius)
    return image

class MyDataset(Dataset):
    def __init__(self, image_dir, img_size, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.listdir = os.listdir(image_dir)
        self.img_size = img_size
        self.radius = img_size[0] // 2

    def __len__(self):
        return len(self.listdir)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.listdir[idx])

        image = cv2.imread(image_path)
        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)
        image = preprocess_image(image, radius=self.radius)
        image1 = random_augmentation(image, radius=self.radius)
        image2 = random_augmentation(image, radius=self.radius)
        image1 = image1 / 255.0
        image2 = image2 / 255.0
        image1 = torch.from_numpy(image1).permute(2, 0, 1)
        image2 = torch.from_numpy(image2).permute(2, 0, 1)

        return image1, image2


def create_dataloader(train_batch_size: int, img_size=(512, 512)):
    train_dataset = MyDataset("./dataset/pretrain/", img_size)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=32, pin_memory=True, drop_last=True)

    return train_loader
