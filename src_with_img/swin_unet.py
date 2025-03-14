import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import swin_tiny_patch4_window7_224 as SwinTransformer

class SwinUNet(nn.Module):
    def __init__(self, img_height=1920, img_width=3840, num_classes=1):
        super(SwinUNet, self).__init__()

        # SwinTransformer Encoder
        self.encoder = SwinTransformer(pretrained=False)

        # Feature extraction layers
        self.enc_out_channels = [96, 192, 384, 768] # output of Swin-Tiny channels

        # Decoder (Unet style upsampling with skip connections)
        self.upconv5 = self._upsample_block(768, 384)
        self.upconv4 = self._upsample_block(384, 192)
        self.upconv3 = self._upsample_block(192, 96)
        self.upconv2 = self._upsample_block(96, 48)
        self.upconv1 = self._upsample_block(48, 24)
        self.final_conv = nn.Conv2d(24, num_classes, kernel_size=1)

    def _upsample_block(self, in_ch, out_ch):
        return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True))

    def forward(self, x):
        B, C, H, W = x.shape

        x_resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)

        # SwinTransformer Forward
        features = self.encoder.forward_features(x_resized) # (B, L, C)

        #Reshape (L = H' * W') -> (B, C, H', W')
        H_new, W_new = H // 32, W // 32 # feature map size of SwinTransformer
        features = features.permute(0, 2, 1).contiguous().view(B, -1, H_new, W_new) # (B, C, H', W')

        # Decoder Path with Skip Connection
        d5 = self.upconv5(features) # (120x240)
        d4 = self.upconv4(d5) # (240x480)
        d3 = self.upconv3(d4) # (480x960)
        d2 = self.upconv2(d3) # (960x1920)
        d1 = self.upconv1(d2) # (1920x3840)
        out = self.final_conv(d1) # (B, 1, 1920, 3840)

        return out
