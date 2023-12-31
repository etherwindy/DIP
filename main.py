import os

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from submit.model import ResNet
from dataloader import create_dataloader
from metric import classification_metrics

# TODO: hyperparameter
EPOCH_NUM = 20
TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 64
lr = 5e-5
weight_decay = 1e-5


def load_weights(model, save_dir: str, epoch: int, device):
    weight_file = os.path.join(save_dir, f"epoch_{epoch}.pth")
    state_dict = torch.load(weight_file, map_location=device)
    state_dict = {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}
    model.load_state_dict(state_dict)


def save_weights(model, save_dir: str, epoch: int):
    weight_file = os.path.join(save_dir, f"epoch_{epoch}.pth")
    torch.save(model.state_dict(), weight_file)


def main():
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)

    writer = SummaryWriter("output")

    model = ResNet()
    backbone = torch.load("pretrainedModel/resnet.pth")
    features = model.backbone.fc.in_features
    backbone.fc = nn.Linear(features, 1)
    model.backbone = backbone
    model.to(device)
    
    criterion = torch.nn.BCELoss()
    train_loader, val_loader = create_dataloader(TRAIN_BATCH_SIZE, VAL_BATCH_SIZE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    for epoch in range(EPOCH_NUM):
        model.train()
        for i, (image, label) in enumerate(train_loader):
            image = image.to(device, torch.float32)
            label = label.to(device, torch.float32)

            output = model(image).squeeze()
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch}/{EPOCH_NUM - 1}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}')
            SummaryWriter.add_scalar(writer, 'train_loss', loss.item(), epoch * len(train_loader) + i)

        print(f"Saving weights of epoch {epoch}...")
        save_weights(model, "output", epoch)

        model.eval()
        with torch.no_grad():
            y_true = []
            y_pred = []
            for i, (image, label) in enumerate(val_loader):
                image = image.to(device, torch.float32)

                output = model(image)
                predict = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).cpu()

                y_true.append(label)
                y_pred.append(predict[:, 0])

            y_true = torch.cat(y_true).numpy()
            y_pred = torch.cat(y_pred).numpy()
            metric = classification_metrics(y_true, y_pred)
            score = (metric["qwk"] + metric["f1"] + metric["spe"]) / 3
            print(metric, score)
            SummaryWriter.add_scalar(writer, 'score', score, epoch * len(train_loader) + i)
    

if __name__ == "__main__":
    main()
