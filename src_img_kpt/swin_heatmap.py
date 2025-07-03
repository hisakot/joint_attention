import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 簡易パッチ分割層 ---
class PatchEmbed(nn.Module):
    def __init__(self, in_chans=5, embed_dim=96, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H/patch, W/patch]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x = self.norm(x)
        return x, H, W

# --- 簡易Swinブロック（ウィンドウAttentionは実装しません）---
class SimpleSwinBlock(nn.Module):
    def __init__(self, dim, num_heads=3, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        # x: [L, B, C] for nn.MultiheadAttention
        shortcut = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = shortcut + x

        shortcut2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut2 + x
        return x

# --- 全体モデル ---
class SimpleSwinHeatmapModel(nn.Module):
    def __init__(self, in_chans=5, embed_dim=96, patch_size=4):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size)
        self.swin_block = SimpleSwinBlock(embed_dim)

        # ヒートマップ用のヘッド（conv層で空間形状に戻すのでembed_dimでOK）
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
        self.patch_size = patch_size

        self.softmax = nn.Softmax()

    def forward(self, x):
        B = x.shape[0]

        # 1. パッチ分割 + 埋め込み
        x, H, W = self.patch_embed(x)  # x: [B, L, C], L=H*W

        # 2. transposeして Attention用に変形 (L, B, C)
        x = x.transpose(0, 1)

        # 3. Swin風ブロック（本家とは異なり単純MultiheadAttention）
        x = self.swin_block(x)

        # 4. (L, B, C) → (B, L, C)
        x = x.transpose(0, 1)

        # 5. ヘッドでチャネルを1に変換
        x = self.head(x)  # [B, L, 1]

        # 6. 空間マップに戻す
        x = x.view(B, H, W)

        # 7. パッチサイズを戻してアップサンプリング
        x = x.unsqueeze(1)  # [B, 1, H, W]
        x = F.interpolate(x, scale_factor=self.patch_size, mode='bilinear', align_corners=False)  # [B, 1, H*patch, W*patch]

        x = self.softmax(x)

        return x

# --- 動作確認 ---
if __name__ == "__main__":
    model = SimpleSwinHeatmapModel(in_chans=5)
    input_tensor = torch.randn(1, 5, 320, 640)
    out = model(input_tensor)
    print(out.shape)  # torch.Size([1, 1, 320, 640])

