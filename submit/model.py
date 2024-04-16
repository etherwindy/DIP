import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as models

from preprocess import preprocess_image


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
        self.model = ResNet()
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
        image = preprocess_image(input_image)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        image = image / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        image = image.to(self.device, torch.float32)

        with torch.no_grad():
            output = self.model(image)
            pred_class = torch.gt(output, 0.5)

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

class VggNet(nn.Module):
    def __init__(self, pretrained=False):

        super(VggNet, self).__init__()
        vgg = models.vgg16(pretrained=pretrained)
        num_features = vgg.classifier[6].in_features
        vgg.classifier[6] = nn.Linear(num_features, 1)
        self.backbone = vgg

    def forward(self, x):
        y = self.backbone(x)
        return torch.sigmoid(y)

class DenseNet(nn.Module):
    def __init__(self, pretrained=False):

        super(DenseNet, self).__init__()
        densenet = models.densenet121(pretrained=pretrained)
        num_features = densenet.classifier.in_features
        densenet.classifier = nn.Linear(num_features, 1)
        self.backbone = densenet

    def forward(self, x):
        y = self.backbone(x)
        return torch.sigmoid(y)