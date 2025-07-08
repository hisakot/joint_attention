import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class SwinTransformerV2B(nn.Module):
    def __init__(self, in_ch=3, img_size=(320, 640)):
        super().__init__()
        self.swin_v2_b = torchvision.models.swin_v2_b()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.swin_v2_b(x)
        print(x.shape)
        exit()

        return x
