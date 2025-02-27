import torch
import torch.nn as nn


class Fusion(nn.Module):
    def __init__(self, in_channels=6, out_channels=3):
        super(Fusion, self).__init__()

        self.fusion_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        concat = torch.cat([x, y], dim=1)

        fused = self.fusion_layer(concat)
        output = self.sigmoid(fused)

        return output
