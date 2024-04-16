import os

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from submit.model import ResNet
from dataloader import create_dataloader
from metric import classification_metrics
from timm.loss import LabelSmoothingCrossEntropy

from mae import models_vit
from mae.util.pos_embed import interpolate_pos_embed
from mae.util.lr_sched import adjust_learning_rate

def get_args_parser():
    parser = argparse.ArgumentParser(description='Train Masked Autoencoder ViT')
    parser.add_argument('--supervise', action="store_true", help='supervised training')
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--nb_classes', default=2, type=int, help='number of classes')
    parser.add_argument('--preprocess', action="store_true", help='preprocess image')
    parser.add_argument('--global_pool', default='avg', help='global pooling')
    parser.add_argument('--drop_path', default=0.1, type=float, help='drop path rate')
    parser.add_argument('--mask_ratio', default=0.75, type=float, help='mask ratio')
    parser.add_argument('--accum_iter', default=1, type=int, help='gradient accumulation steps')
    parser.add_argument('--warmup_epochs', default=2, type=int, help='warmup epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--val_batch_size', default=64, type=int, help='val batch size')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=1e-6, type=float, help='minimum learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--bce', action="store_true", help='use BCE loss')
    parser.add_argument('--smoothing', default=0, type=float, help='label smoothing')
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


def main(args):
    if not torch.cuda.is_available():
        print("Cuda is not available")
        exit()
        
    device = torch.device('cuda', args.gpu)

    if not os.path.exists("output"):
        os.mkdir("output")

    writer = SummaryWriter("output/vit")

    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )
    
    if args.supervise is False:
        model_state = torch.load("pretrainedModel/vit_base/vit.pth")
        interpolate_pos_embed(model, model_state)
        msg = model.load_state_dict(model_state, strict=False)
        print(msg)
    
    model.to(device)
    
    if args.bce:
        assert args.nb_classes == 1
        criterion = torch.nn.BCELoss()
    elif args.smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    train_loader, val_loader = create_dataloader(args.batch_size, args.val_batch_size, img_size=(224, 224), preprocess=args.preprocess)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    for epoch in range(args.epochs):
        model.train()
        for i, (image, label) in enumerate(train_loader):
            adjust_learning_rate(optimizer, i / len(train_loader) + epoch, args)
            image = image.to(device, torch.float32)
            output = model(image)
            if args.bce:
                label = label.to(device, torch.float32)
                output = torch.sigmoid(output).squeeze()
                acc = torch.sum(torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)) == label).item() / len(label)
            else:
                label = label.to(device, torch.long)
                F.one_hot(label, num_classes=args.nb_classes)
                acc = torch.sum(torch.argmax(output, dim=1) == label).item() / len(label)
            loss = criterion(output, label) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch}/{args.epochs - 1}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}, Acc: {acc}')
            SummaryWriter.add_scalar(writer, 'train_loss', loss.item(), epoch * len(train_loader) + i)

        print(f"Saving weights of epoch {epoch}...")
        save_weights(model, "output/vit", epoch)

        model.eval()
        with torch.no_grad():
            y_true = []
            y_pred = []
            for i, (image, label) in enumerate(val_loader):
                image = image.to(device, torch.float32)

                output = model(image)
                if args.bce:
                    output = torch.sigmoid(output).squeeze()
                else:
                    output = F.softmax(output, dim=1)[:, 1]
                predict = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).cpu()

                y_true.append(label)
                y_pred.append(predict)

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
