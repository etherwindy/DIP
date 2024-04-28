import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as models

from preprocess import preprocess_image
from tools import partial

import timm.models.vision_transformer


class model:
    def __init__(self):
        self.device = torch.device("cpu")

    def load(self, dir_path):
        """
        load the model and weights.
        dir_path is a string for internal use only - do not remove it.
        all other paths should only contain the file name, these paths must be
        concatenated with dir_path, for example: os.path.join(dir_path, filename).
        make sure these files are in the same directory as the model.py file.
        :param dir_path: path to the submission directory (for internal use only).
        :return:
        """
        self.model1 = ResNet34()
        #self.model2 = DenseNet161()
        self.model3 = vit_base_patch16(
            num_classes=1,
            drop_path_rate=0.1,
            global_pool='avg',
        )
        #self.model4 = ResNet50()
        self.model5 = DenseNet121()
        checkpoint_path1 = os.path.join(dir_path, "resnet34.pth")
        #checkpoint_path2 = os.path.join(dir_path, "densenet161.pth")
        checkpoint_path3 = os.path.join(dir_path, "vit_base.pth")
        #checkpoint_path4 = os.path.join(dir_path, "resnet50.pth")
        checkpoint_path5 = os.path.join(dir_path, "densenet121.pth")
        self.model1.load_state_dict(torch.load(checkpoint_path1, map_location=self.device))
        #self.model2.load_state_dict(torch.load(checkpoint_path2, map_location=self.device))
        self.model3.load_state_dict(torch.load(checkpoint_path3, map_location=self.device))
        #self.model4.load_state_dict(torch.load(checkpoint_path4, map_location=self.device))
        self.model5.load_state_dict(torch.load(checkpoint_path5, map_location=self.device))
        self.model1.to(self.device)
        self.model1.eval()
        #self.model2.to(self.device)
        #self.model2.eval()
        self.model3.to(self.device)
        self.model3.eval()
        #self.model4.to(self.device)
        #self.model4.eval()
        self.model5.to(self.device)
        self.model5.eval()

    def predict(self, input_image):
        """
        perform the prediction given an image.
        input_image is a ndarray read using cv2.imread(path_to_image, 1).
        note that the order of the three channels of the input_image read by cv2 is BGR.
        :param input_image: the input image to the model.
        :return: an int value indicating the class for the input image
        """
        image = preprocess_image(input_image)
        image1 = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        image2 = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        image3 = cv2.resize(input_image, (224, 224), interpolation=cv2.INTER_LINEAR)
        image1 = image1 / 255.0
        image2 = image2 / 255.0
        image1 = torch.from_numpy(image1).permute(2, 0, 1).unsqueeze(0)
        image2 = torch.from_numpy(image2).permute(2, 0, 1).unsqueeze(0)
        image3 = torch.from_numpy(image3).permute(2, 0, 1).unsqueeze(0)
        image1 = image1.to(self.device, torch.float32)
        image2 = image2.to(self.device, torch.float32)
        image3 = image3.to(self.device, torch.float32)

        with torch.no_grad():
            output1 = self.model1(image1)
            #output2 = self.model2(image2)
            output3 = self.model3(image3)
            #output4 = self.model4(image2)
            output5 = self.model5(image2)
            pred_class1 = torch.gt(output1, 0.45)
            #pred_class2 = torch.gt(output2, 0.55)
            pred_class3 = torch.gt(output3, 0.5)
            #pred_class4 = torch.gt(output4, 0.5)
            pred_class5 = torch.gt(output5, 0.5)
            #pred_class3 = torch.argmax(output3, dim=1)

        pred_class1 = int(pred_class1.detach().cpu())
        #pred_class2 = int(pred_class2.detach().cpu())
        pred_class3 = int(pred_class3.detach().cpu())
        #pred_class4 = int(pred_class4.detach().cpu())
        pred_class5 = int(pred_class5.detach().cpu())
        sum = pred_class1 + pred_class3 + pred_class5
        #if sum >= 2:
        #    return 1
        #else:
        #    return 0
        return pred_class5

class ResNet34(nn.Module):
    def __init__(self, pretrained=False):

        super(ResNet34, self).__init__()
        resnet = models.resnet34(pretrained=pretrained)
        num_features = resnet.fc.in_features
        resnet.fc = nn.Linear(num_features, 1)
        self.backbone = resnet

    def forward(self, x):
        y = self.backbone(x)
        return torch.sigmoid(y)

class ResNet50(nn.Module):
    def __init__(self, pretrained=False):

        super(ResNet50, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        num_features = resnet.fc.in_features
        resnet.fc = nn.Linear(num_features, 1)
        self.backbone = resnet

    def forward(self, x):
        y = self.backbone(x)
        return torch.sigmoid(y)

class DenseNet121(nn.Module):
    def __init__(self, pretrained=False):

        super(DenseNet121, self).__init__()
        densenet = models.densenet121(pretrained=pretrained)
        num_features = densenet.classifier.in_features
        densenet.classifier = nn.Linear(num_features, 1)
        self.backbone = densenet

    def forward(self, x):
        y = self.backbone(x)
        return torch.sigmoid(y)

class DenseNet161(nn.Module):
    def __init__(self, pretrained=False):

        super(DenseNet161, self).__init__()
        densenet = models.densenet161(pretrained=pretrained)
        num_features = densenet.classifier.in_features
        densenet.classifier = nn.Linear(num_features, 1)
        self.backbone = densenet

    def forward(self, x):
        y = self.backbone(x)
        return torch.sigmoid(y)

class EfficientNet(nn.Module):
    def __init__(self, pretrained=False):
        
        super(EfficientNet, self).__init__()
        efficientnet = models.efficientnet_v2_s(pretrained=pretrained)
        num_features = efficientnet.classifier[1].in_features
        efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(in_features=num_features, out_features=1)
            )
        self.backbone = efficientnet
    
    def forward(self, x):
        y = self.backbone(x)
        return torch.sigmoid(y)

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        #if self.global_pool:
        #    x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        #    outcome = self.fc_norm(x)
        #else:
        #    x = self.norm(x)
        #    outcome = x[:, 0]

        #return outcome
        return x

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model