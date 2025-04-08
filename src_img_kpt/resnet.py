import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F

import config

model_urls = {
        "resnet50" : "https://download.pytorch.org/models/resnet50-19c8e357.pth",
        }

class ResNet50(nn.Module):
    def __init__(self, pretrained, in_ch, num_output=3*100*200):
        super(ResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=pretrained, num_classes=3000)
        self.resnet50.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2,
                                        padding=3, bias=False)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_output)

        cfg = config.Config()
        self.img_H = cfg.img_height
        self.img_W = cfg.img_width

    def forward(self, x):
        x = self.resnet50(x)
        x = F.sigmoid(x)
        x = x.reshape(-1, 3, 100, 200)
        x = F.interpolate(x, (self.img_H, self.img_W), mode="bilinear", align_corners=False)

        return x
