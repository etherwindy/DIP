import os

import math
import numpy as np
import torch
import argparse
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

def get_args_parser():
    parser = argparse.ArgumentParser(description='Train Masked Autoencoder ViT')
    parser.add_argument('--img_size', default=(512, 512), type=tuple, help='image size')
    parser.add_argument('--warmup_epochs', default=4, type=int, help='warmup epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--val_batch_size', default=64, type=int, help='val batch size')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=1e-5, type=float, help='minimum learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    return parser


def load_weights(model, save_dir: str, epoch: int, device):
    weight_file = os.path.join(save_dir, f"epoch_{epoch}.pth")
    state_dict = torch.load(weight_file, map_location=device)
    state_dict = {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}
    model.load_state_dict(state_dict)


def save_weights(model, save_dir: str, epoch: int):
    weight_file = os.path.join(save_dir, f"epoch_{epoch}.pth")
    torch.save(model.state_dict(), weight_file)

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def main(args):
    if not torch.cuda.is_available():
        print("Cuda is not available")
        exit()
        
    device = torch.device('cuda', args.gpu)

    if not os.path.exists("output"):
        os.mkdir("output")

    writer = SummaryWriter("output/resnet")

    model = ResNet()
    backbone = torch.load("pretrainedModel/resnet/resnet.pth")
    features = model.backbone.fc.in_features
    backbone.fc = nn.Linear(features, 1)
    model.backbone = backbone
    model.to(device)
    
    criterion = torch.nn.BCELoss()
    train_loader, val_loader = create_dataloader(args.batch_size, args.val_batch_size, img_size=args.img_size, preprocess=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    
    for epoch in range(EPOCH_NUM):
        model.train()
        for i, (image, label) in enumerate(train_loader):
            adjust_learning_rate(optimizer, i / len(train_loader) + epoch, args)
            
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
        save_weights(model, "output/resnet", epoch)

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
    args = get_args_parser()
    args = args.parse_args()
    main(args)
