import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.model = vit_base_patch16(
            num_classes=2,
            drop_path_rate=0.1,
            global_pool='avg',
        )
        checkpoint_path = os.path.join(dir_path, "model_weights.pth")
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, input_image):
        """
        perform the prediction given an image.
        input_image is a ndarray read using cv2.imread(path_to_image, 1).
        note that the order of the three channels of the input_image read by cv2 is BGR.
        :param input_image: the input image to the model.
        :return: an int value indicating the class for the input image
        """
        #image = preprocess_image(input_image)
        #image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        #image = image / 255.0
        #image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        #image = image.to(self.device, torch.float32)

        input_image = cv2.resize(input_image, (224, 224), interpolation=cv2.INTER_LINEAR)
        input_image = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0)
        input_image = input_image.to(self.device, torch.float32)

        with torch.no_grad():
            output = self.model(input_image).squeeze()
            output = F.softmax(output, dim=0)
            pred_class = torch.gt(output[1], 0.5)

        pred_class = int(pred_class.detach().cpu())
        return pred_class


class ResNet(nn.Module):
    def __init__(self, pretrained=False):

        super(ResNet, self).__init__()
        resnet = models.resnet34(pretrained=pretrained)
        num_features = resnet.fc.in_features
        resnet.fc = nn.Linear(num_features, 1)
        self.backbone = resnet

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


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

#if __name__ == "__main__":
#    model = model()
#    img = "../dataset/image_original/0000a3ac.png"
#    model.load("./")
#    predict=model.predict(cv2.imread(img, 1))
#    print(predict)
    