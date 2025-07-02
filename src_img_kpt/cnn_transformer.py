import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.models import resnet18

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(in_channels, in_channels // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels // reduction, in_channels),
                nn.Sigmoid()
                )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class CNNBackbone(nn.Module):
    def __init__(self, in_ch=3, out_channels=256):
        super().__init__()
        resnet = resnet18(pretrained=False)
        self.feature_extractor = nn.Sequential(
                # resnet.conv1, # (B, 64, H/2, W/2)
                nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
                resnet.bn1,
                resnet.relu,
                # resnet.maxpool, # (B, 64, H/4, W/4) # TODO if output (224, 224)
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # TODO if output(224, 448)
                resnet.layer1, # (B, 64, H/4, W/4)
                resnet.layer2, # (B, 128, H/8, W/8)
                resnet.layer3, # (B, 256, H/16, W/16)
                )
        self.se = SEBlock(256)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.se(features)
        return features

class CNN2TransformerAdapter(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.conv1x1 = nn.Conv2d(256, embed_dim, kernel_size=1)

    def forward(self, x):
        x = self.conv1x1(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # (B, HW, C)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=256, num_layers=4, num_head=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim, nhead=num_head, batch_first=True, dropout=0.3)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.encoder(x)

class TransformerHead(nn.Module):
    def __init__(self, embed_dim=256, out_size=(20, 40)):
        super().__init__()
        self.output_size = out_size
        self.mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, 1)
                )
    
    def forward(self, x):
        x = self.mlp(x).squeeze(-1) # (B, HW)
        B = x.size(0)
        x = x.view(B, 1, *self.output_size)
        return x

class CNNTransformer2Heatmap(nn.Module):
    def __init__(self, in_channels=3, img_size=(320, 640), output_size=(480, 960)):
        super().__init__()
        self.img_size = img_size
        self.output_size = output_size
        self.backbone = CNNBackbone(in_ch=in_channels)
        self.adapter = CNN2TransformerAdapter(embed_dim=256)
        self.decoder = TransformerDecoder(embed_dim=256)
        self.head =TransformerHead(embed_dim=256, out_size=output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x) # (B, 512, H/8, W/8)
        x, hw = self.adapter(x) # (B, HW, 256)
        x = self.decoder(x) # (B, 1, H/16, W/16)
        x = self.head(x)
        x = self.sigmoid(x)
        x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False) # (B, 1, 320, 640)

        return x

