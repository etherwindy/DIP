import os
import torch
import torch.nn as nn
import torchvision
import argparse
from torchvision.models import resnet34
from mae import dataloader
from mae import trainer
from mae import models_mae


def get_args_parser():
    parser = argparse.ArgumentParser(description='pretain ViT')
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--mask_ratio', default=0.75, type=float, help='mask ratio')
    parser.add_argument('--accum_iter', default=1, type=int, help='gradient accumulation steps')
    parser.add_argument('--warmup_epochs', default=10, type=int, help='warmup epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=1e-5, type=float, help='min learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--gpu', default=0, type=int, help='gpu number')
    return parser

def main(args):
    if not os.path.exists('pretrainedModel'):
        os.mkdir('pretrainedModel')
    dir = 'pretrainedModel/vit'
    if not os.path.exists(dir):
        os.mkdir(dir)
    
    if not torch.cuda.is_available():
        print("Cuda is not available")
        exit()
        
    device = torch.device('cuda', args.gpu)
    
    train_loader = dataloader.create_dataloader(args.batch_size)
    
    vit = models_mae.__dict__[args.model]()
    vit = vit.to(device)
    
    
    trainer.mae_trainer(
        model=vit,
        train_loader=train_loader,
        num_epochs=args.epochs,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        args=args
    )
    
    vit.eval()
    path = os.path.join(dir, 'vit.pth')
    torch.save(vit.state_dict(), path)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)