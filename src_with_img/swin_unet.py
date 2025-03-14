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
        self.upconv3 = self._upsample_block(self.enc_out_channels[3], self.enc_out_channels[2])
        self.upconv2 = self._upsample_block(self.enc_out_channels[2], self.enc_out_channels[1])
        self.upconv1 = self._upsample_block(self.enc_out_channels[1], self.enc_out_channels[0])
        self.final_conv = nn.Conv2d(self.enc_out_channels[0], num_classes, kernel_size=1)

    def _upsample_block(self, in_ch, out_ch):
        return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

    def forward(self, x):
        B, C, H, W = x.shape
        # SwinTransformer Forward
        features = self.encoder.forward_features(x) # (B, L, C)

        #Reshape (L = H' * W') -> (B, C, H', W')
        H_new, W_new = H // 32, W // 32 # feature map size of SwinTransformer
        features = features.permute(0, 2, 1).contiguous().view(B, -1, H_new, W_new) # (B, C, H', W')

        # Decoder Path with Skip Connection
        d3 = self.upconv3(features) # B, 384, Hx2, Wx2)
        d2 = self.upconv2(d3) # B, 192, Hx4, Wx4)
        d1 = self.upconv1(d2) # B, 96, Hx8, Wx8)
        out = self.final_conv(d1) # B, 1, Hx8, Wx8)

        return out
