import os
import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet34
from simclr import dataloader
from simclr import trainer
from torch.utils.data import DataLoader


batch_size = 64
epochs = 5
temperature = 0.07
lr = 1e-4
weight_decay = 1e-5


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def main():
    dir = 'pretrainedModel'
    if not os.path.exists(dir):
        os.mkdir(dir)
    
    if not torch.cuda.is_available():
        print("Cuda is not available")
        exit()

    device = torch.device('cuda')

    train_loader = dataloader.create_dataloader(batch_size)

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
        num_epochs=epochs,
        temperature=2,
        device=device,
        learning_rate=lr,
        weight_decay=weight_decay,
    )

    resnet.eval()
    path = os.path.join(dir, 'resnet.pth')
    torch.save(resnet, path)


if __name__ == '__main__':
    main()
