import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size, img_H, img_W):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.projection = nn.Linear(patch_size * patch_size * in_channels, emb_size)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.position_embeddings = nn.Parameter(torch.randn((img_H // patch_size) * (img_W // patch_size), emb_size))

    def forward(self, x):
        b, c, h, w = x.shape
        p = self.patch_size

        x = x.view(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(b, -1, p * p * c)

        # Patch to embedding
        x = self.projection(x)

        '''
        # add CLS token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        '''

        # add posiotional encoding
        x += self.position_embeddings
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.unify_heads = nn.Linear(emb_size, emb_size)

    def forward(self, x):
        b, t, e = x.size()
        h = self.num_heads
        assert e == self.emb_size, f'Expected input embedding size {self.emb_size}, but got {e}'

        # each head
        keys = self.keys(x).view(b, t, h, e // h).transpose(1, 2)
        queries = self.queries(x).view(b, t, h, e // h).transpose(1, 2)
        values = self.values(x).view(b, t, h, e // h).transpose(1, 2)

        # attention
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        attention = torch.softmax(energy / (e ** (1 / 2)), dim=-1)

        out = torch.einsum('bhqk, bhkd -> bhqd', attention, values).transpose(1, 2).contiguous()
        out = out.view(b, t, e)
        return self.unify_heads(out)

class TransformerBlock(nn.Module):
    def __init__(self, emb_size, num_heads, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(emb_size, num_heads)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)

        #MLP
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, forward_expansion * emb_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * emb_size, emb_size)
        )

    def forward(self, x):
        x = self.attention(self.norm1(x)) + x
        x = self.feed_forward(self.norm2(x)) + x
        return x

class Upsampling(nn.Module):
    def __init__(self):
        super(Upsampling, self).__init__()
        self.pixshuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        B, L, C = x.shape
        re_size = int(math.sqrt(L // 2))
        x = x.permute(0, 2, 1)
        x = torch.reshape(x, (B, C, re_size, re_size * 2))
        x = self.pixshuffle(x)
        B, C, H, W = x.shape
        x = torch.reshape(x, (B, C, H * W))
        x = x.permute(0, 2, 1)
        return x

class UpSampleBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_dim), 
                nn.ReLU(inplace=True)
                )

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.block(x)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(1, 2).contiguous()
        return x, H, W

class UpConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
                )

    def forward(self, x):
        B, L, C = x.shape
        height = int(math.sqrt(L // 2))
        width = int(height * 2)
        x = torch.reshape(x, (B, height, width, 1)) # (B, 100, 200, 1)
        x = x.permute(0, 3, 1, 2) # (B, 1, 100, 200)
        x = self.up(x)
        x = self.conv(x)
        return x

class FinalConv(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.out = nn.Sequential(
                nn.Conv2d(in_ch, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1),
                nn.Sigmoid()
                )

    def forward(self, x):
        return self.out(x)


'''
class VisionTransformer(nn.Module):
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes*2)
        )
'''

class TransGAN(nn.Module):
    def __init__(self, patch_size, emb_size, num_heads, forward_expansion, img_height, img_width, in_ch):
        super(TransGAN, self).__init__()
        self.img_height = img_height
        self.img_width = img_width

        # Downsample input for efficiency
        self.input_size = (100, 200)  # fixed size for patch embedding
        self.embedding = PatchEmbedding(in_ch, patch_size, emb_size, *self.input_size)

        # Transformer blocks (same emb_size throughout)
        self.encoder1 = TransformerBlock(emb_size, num_heads, forward_expansion)
        self.encoder2 = TransformerBlock(emb_size, num_heads, forward_expansion)
        self.encoder3 = TransformerBlock(emb_size, num_heads, forward_expansion)

        # Upsample blocks
        self.upsample1 = UpSampleBlock(emb_size, emb_size // 2)
        self.upsample2 = UpSampleBlock(emb_size // 2, emb_size // 4)

        # Final conv layers
        self.final_conv = FinalConv(emb_size // 4)

    def forward(self, x):
        B, H, W, C = x.shape
        x = F.interpolate(x, self.input_size, mode='bilinear', align_corners=False)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

        # Patch Embedding
        x = self.embedding(x)  # (B, L, emb_size)

        # Transformer
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)

        # Upsample 1
        x, h1, w1 = self.upsample1(x, H=10, W=20)  # H=100/10, W=200/10
        x, h2, w2 = self.upsample2(x, H=h1, W=w1)

        # Final conv
        x = x.transpose(1, 2).contiguous().view(B, -1, h2, w2)
        x = self.final_conv(x)

        # Upsample to original
        x = F.interpolate(x, (self.img_height, self.img_width), mode='bilinear', align_corners=False)
        return x

class TransGAN_v1(nn.Module):
    def __init__(self, patch_size, emb_size, num_heads, forward_expansion, img_height, img_width, in_ch):
        super(TransGAN, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.embedding = PatchEmbedding(in_ch, patch_size, emb_size, 100, 200)
        self.encoder1 = TransformerBlock(emb_size, num_heads, forward_expansion)
        self.encoder2 = TransformerBlock(emb_size // 4, num_heads, forward_expansion)
        self.encoder3 = TransformerBlock(emb_size // (4**2), num_heads, forward_expansion)
        self.upsampling = Upsampling()
        self.upconv = UpConvBlock(1, 1)
        self.fc = nn.Linear(emb_size // (4**2), 1) # TODO
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, H, W, C = x.shape
        x = F.interpolate(x, (100, 200), mode='bilinear', align_corners=False)
        x = self.embedding(x) # (B, 200, 256)
        x = self.encoder1(x) # (B, 200, 256)
        x = self.upsampling(x) # (B, 800, 64)
        x = self.encoder2(x) # (B, 800, 64)
        x = self.upsampling(x) # (B, 3200, 16)
        x = self.encoder3(x) # (B, 3200, 16)
        x = self.fc(x) # (B, 3200, 1)
        x = self.upconv(x) # (B, 1, 160, 320)
        x = self.sigmoid(x)
        x = F.interpolate(x, (self.img_height, self.img_width), mode='bilinear', align_corners=False)
        return x
