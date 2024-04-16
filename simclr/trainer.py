import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import math

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

class NTXentLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z):
        # 将每个样本的特征向量 z1 和 z2 连接起来形成正样本对
        # 正样本对是由同一个输入样本生成的两个样本
        # 例如：在图像数据中，可能是由同一张图片经过不同的数据增强而得到的两个版本
        # 输入参数 z 已经将z1 和 z2 相连
        z = F.normalize(z, dim=1)
        batch_size = z.size(0) // 2

        # 计算所有样本之间的余弦相似度
        sim_matrix = torch.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        # 构建匹配（positives）和不匹配（negatives）样本对的标签
        # 对于每个样本，除了正样本对，其余样本都认为是负样本
        mask = torch.eye(2*batch_size, device=z.device) > 0.5
        positives_ij = torch.diag(sim_matrix, batch_size).view(batch_size, -1)
        positives_ji = torch.diag(sim_matrix, -batch_size).view(batch_size, -1)
        positives = torch.cat([positives_ij, positives_ji], dim=0)
        negatives = sim_matrix[~mask].view(2*batch_size, -1)

        # 计算正样本对的相似度以及负样本对的相似度
        logits = torch.cat([positives, negatives], dim=1) / self.temperature

        # 计算 NT-Xent Loss
        labels = torch.zeros(logits.shape[0], device=z.device, dtype=torch.long)
        loss = F.cross_entropy(logits, labels)

        return loss


def contrastive_trainer(model, train_loader, num_epochs, temperature, device, learning_rate=0.001, weight_decay=1e-5, args=None):
    writer = SummaryWriter("pretrainedModel/resnet")
    criterion = NTXentLoss(temperature)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if isinstance(model, torch.nn.Module):
        model.train()

    for epoch in range(num_epochs):
        for i, (x1, x2) in enumerate(train_loader):
            adjust_learning_rate(optimizer, i / len(train_loader) + epoch, args)
            
            x1 = x1.to(device, torch.float32)
            x2 = x2.to(device, torch.float32)
            x = torch.cat([x1, x2], dim=0)
            optimizer.zero_grad()
            z = model(x)

            loss = criterion(z)
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch}/{num_epochs-1}] Step [{i+1}/{len(train_loader)}] "
                  f"Loss: {loss.item()}")
            SummaryWriter.add_scalar(writer, 'train_loss', loss.item(), epoch * len(train_loader) + i)