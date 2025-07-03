import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SwinHeatmapModel(nn.Module):
    def __init__(self, in_ch, backbone_name='swin_tiny_patch4_window7_224'):
        super().__init__()
        
        # 1. Swin Transformer のバックボーン（特徴抽出）
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, features_only=False, in_chans=in_ch
        )
        self.out_channels = self.backbone.feature_info.channels()  # [96, 192, 384, 768]
        
        # 2. Lateral convs (FPNスタイルでチャネル整形)
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, 128, kernel_size=1) for c in self.out_channels
        ])
        
        # 3. 統合後の3x3 Conv
        self.fuse_conv = nn.Conv2d(128 * 4, 128, kernel_size=3, padding=1)
        
        # 4. 出力ヘッド（1チャンネルのヒートマップ）
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)  # 1チャンネル出力
        )

    def forward(self, x):
        input_shape = x.shape[2:]  # 元画像サイズ

        # 特徴抽出（Stage1〜Stage4）
        feats = self.backbone(x)  # list of [B, C, H, W]
        print("backbone: ", feats.shape)

        # Lateral convs
        feats = [lat(f) for lat, f in zip(self.laterals, feats)]

        # 各特徴を最大解像度にアップサンプリング（C1と同じにする）
        target_size = feats[0].shape[2:]
        feats_up = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False) for f in feats]

        # 結合 → Conv → Heatmap
        fused = torch.cat(feats_up, dim=1)  # [B, 128*4, H, W]
        fused = self.fuse_conv(fused)      # [B, 128, H, W]
        heatmap = self.heatmap_head(fused) # [B, 1, H, W]
        print("heatmap: ", heatmap.shape)

        # 最後に元画像サイズにアップサンプリング
        heatmap = F.interpolate(heatmap, size=input_shape, mode='bilinear', align_corners=False)
        return heatmap

