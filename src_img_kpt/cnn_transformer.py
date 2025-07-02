import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class CNNBackbone(nn.Module):
    def __init__(self, in_ch=3, out_channels=256):
        super().__init__()
        resnet = resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(
                # resnet.conv1, # (B, 64, H/2, W/2)
                nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
                resnet.bn1,
                resnet.relu,
                # resnet.maxpool, # (B, 64, H/4, W/4) # TODO if output (224, 224)
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # TODO id output(224, 448)
                resnet.layer1, # (B, 256, H/4, W/4)
                resnet.layer2, # (B, 512, H/8, W/8)
                # resnet.layer3, # (B, 1024, H/16, W/16)
                )

    def forward(self, x):
        return self.feature_extractor(x)

class CNN2TransformerAdapter(nn.Module):
    def __init__(self, embed_dim=512, max_hw=128*256):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_hw, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.max_hw = max_hw

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # (B, HW, C)

        self.pos_embed = self.pos_embed[:, :H*W, :]
        x = x + pos_embed # adding positional information
        return x, (H, W)

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=512, num_layers=4, num_head=8):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_head, batch_first=True, dropout=0.1),
                num_layers=num_layers
                )
        self.head = nn.Linear(embed_dim, 1) # for heatmap

    def forward(self, x, hw):
        x = self.transformer(x) # (B, HW, C)
        x = self.head(x) # (B, HW, 1)
        B, HW, _ = x.shape
        H, W = hw
        x = x.permute(0, 2, 1).reshape(B, 1, H, W) # (B, 1, H, W)
        return x

class CNNTransformer2Heatmap(nn.Module):
    def __init__(self, in_channels=3, img_size=(320, 640), output_size=(480, 960)):
        super().__init__()
        self.img_size = img_size
        self.output_size = output_size
        self.backbone = CNNBackbone(in_ch=in_channels)
        self.adapter = CNN2TransformerAdapter(embed_dim=512)
        self.decoder = TransformerDecoder(embed_dim=512)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x) # (B, 512, H/8, W/8)
        x, hw = self.adapter(x) # (B, HW, 1024)
        x = self.decoder(x, hw) # (B, 1, H/16, W/16)
        x = self.sigmoid(x)
        x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False) # (B, 1, 320, 640)

        # x = F.interpolate(output, size=(1920, 3840), mode='bilinear', align_corners=False)
        return x

