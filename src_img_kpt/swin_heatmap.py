import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SwinHeatmapModel(nn.Module):
    def __init__(self, backbone_name='swin_tiny_patch4_window7_224', in_chans=3):
        super().__init__()
        # Swin Transformer backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            features_only=False,   # ← ここがポイント
            in_chans=in_chans
        )
        self.norm = self.backbone.norm  # LayerNorm
        self.patch_size = 4  # swin_tiny_patch4_window7_224 はパッチサイズ4

        # ヒートマップを生成するためのヘッド
        # Swinの出力チャネル数（768）に合わせる
        self.head = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1)
        )

        self.softmax = nn.Softmax()

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. Swin の特徴抽出（features_only=False時はflatten出力）
        x = self.backbone.forward_features(x)  # [B, L, C_feat]  (L=patch数)

        # 2. LayerNorm（backboneの最後のNormを通す）
        x = self.norm(x)  # [B, L, C_feat]

        # 3. パッチ数から空間サイズに変換
        L, C_feat = x.shape[1], x.shape[2]
        h_feat, w_feat = H // self.patch_size, W // self.patch_size
        x = x.transpose(1, 2).reshape(B, C_feat, h_feat, w_feat)  # [B, C, H_patch, W_patch]

        # 4. ヒートマップヘッド
        heatmap = self.head(x)  # [B, 1, H_patch, W_patch]

        # 5. 元画像サイズにアップサンプリング
        heatmap = F.interpolate(heatmap, size=(H, W), mode='bilinear', align_corners=False)

        heatmap = self.softmax(heatmap)

        return heatmap

