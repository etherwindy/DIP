import os
import torch
import torch.nn as nn
import torchvision
import argparse
from torchvision.models import resnet34
from simclr import dataloader
from simclr import trainer
from torch.utils.data import DataLoader


def get_args_parser():
    parser = argparse.ArgumentParser(description='Pretrain ResNet')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--temperature', default=0.07, type=float, help='temperature')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=1e-5, type=float, help='minimum learning rate')
    parser.add_argument('--warmup_epochs', default=10, type=int, help='number of warmup epochs')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--img_size', default=(512, 512), type=tuple, help='image size')
    parser.add_argument('--gpu', default=0, type=int, help='gpu number')
    return parser

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def main(args):
    if not os.path.exists('pretrainedModel'):
        os.mkdir('pretrainedModel')
    dir = 'pretrainedModel/resnet'
    if not os.path.exists(dir):
        os.mkdir(dir)
    
    if not torch.cuda.is_available():
        print("Cuda is not available")
        exit()

    device = torch.device('cuda', args.gpu)

    train_loader = dataloader.create_dataloader(args.batch_size, img_size=args.img_size)

    resnet = resnet34()
    features = resnet.fc.in_features
    resnet.fc = nn.Sequential(
                    nn.Linear(features, features),
                    nn.ReLU(),
                    nn.Linear(features, features)
                )
    resnet = resnet.to(device)

    trainer.contrastive_trainer(
        model=resnet,
        train_loader=train_loader,
        num_epochs=args.epochs,
        temperature=args.temperature,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        args=args
    )

    resnet.eval()
    path = os.path.join(dir, 'resnet.pth')
    torch.save(resnet, path)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
