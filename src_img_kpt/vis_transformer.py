import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size, img_H, img_W):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.projection = nn.Linear(patch_size * patch_size * in_channels, emb_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.position_embeddings = nn.Parameter(torch.randn((img_H // patch_size) * (img_W // patch_size) + 1, emb_size))

    def forward(self, x):
        b, c, h, w = x.shape
        p = self.patch_size

        x = x.view(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(b, -1, p * p * c)

        # Patch to embedding
        x = self.projection(x)

        # add CLS token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

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

class VisionTransformer(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size, img_H, img_W, num_layers, num_heads, forward_expansion, num_classes):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_H, img_W)
        self.transformer = nn.Sequential(
            *[TransformerBlock(emb_size, num_heads, forward_expansion) for _ in range(num_layers)]
        )
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes*2)
        )

        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=0)
        self.deconv_bn1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0)
        self.deconv_bn2 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0)
        self.deconv_bn3 = nn.BatchNorm2d(32)
        self.deconv4 = nn.ConvTranspose2d(32, 8, kernel_size=3, stride=4, padding=0)
        self.deconv_bn4 = nn.BatchNorm2d(8)
        self.deconv5 = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=4, padding=0)
        self.deconv_bn5 = nn.BatchNorm2d(1)
        self.conv = nn.Conv2d(1, 1, kernel_size=1, stride=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])
        x = self.mlp_head(x)
        b, num_cls = x.shape
        x = x.view(b, int(num_cls/2), 1, 2)
        x = self.deconv1(x)
        x = self.deconv_bn1(x)
        x = self.deconv2(x)
        x = self.deconv_bn2(x)
        x = self.deconv3(x)
        x = self.deconv_bn3(x)
        x = self.deconv4(x)
        x = self.deconv_bn4(x)
        x = self.deconv5(x)
        x = self.deconv_bn5(x)
        x = self.conv(x)
        x = self.sigmoid(x)

        x = F.interpolate(x, (320, 640), mode='bilinear')
        return x
