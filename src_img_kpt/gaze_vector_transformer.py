import torch
import torch.nn as nn

class PatchToImage(nn.Module):
    def __init__(self, patch_size, image_height, image_width, channels):
        super().__init__()
        self.patch_size = patch_size
        self.image_height = image_height
        self.image_width = image_width
        self.channels = channels

    def forward(self, x):
        # x: [B, N, patch_dim] where patch_dim = C * P * P
        B, N, D = x.shape
        P = self.patch_size
        H = self.image_height // P
        W = self.image_width // P
        x = x.view(B, H, W, self.channels, P, P)
        x = x.permute(0, 3, 1, 4, 2, 5) # [B, C, H, P, W, P]
        x = x.reshape(B, self.channels, self.image_height, self.image_width)
        return x

class VariableLengthVectorToImage(nn.Moduke):
    def __init__(self, input_dim, image_height=320, image_width=640, patch_size=8, embed_dim=256, num_heads=4, num_layers=6):
        super().__init__()

        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1024, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transforner = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.image_height = image_height
        self.image_width = image_width
        self.patch_size = patch_size
        self.num_patches = (image_height // patch_size) * (image_width // patch_size)
        self.project = nn.Linear(embed_dim, 3 * patch_size * patch_size)
        self.unpatchfy = PatchToImage(patch_size, image_height, image_width, channels=3)

    def forward(self, x): # x: [B, L, input_dim]
        B, L, _ = x.shape
        x = self.embedding(x) + self.pos_encoding[:L] # {B, L, D]
        x = x.permute(1, 0, 2) # [L, B, D] -> Transformer expects time first
        x = self.transformer(x)
        x = x.permute(1, 0, 2) # [B, L, D]

        pooled = x.mean(dim=1)
        expanded = pooled.unsqueeze(1).repeat(1, self.num_patches, 1) # [B, N, D]

        patches = self.project(expanded) # [B, N, patch_dim]
        img = self.unpatchfy(patches)

        return img
