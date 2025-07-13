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
        self.embedding = PatchEmbedding(in_ch, patch_size, emb_size, img_height, img_width)
        self.encoder1 = TransformerBlock(emb_size, num_heads, forward_expansion)
        self.encoder2 = TransformerBlock(emb_size // 4, num_heads, forward_expansion)
        self.encoder3 = TransformerBlock(emb_size // (4**2), num_heads, forward_expansion)
        self.upsampling = Upsampling()
        self.fc = nn.Linear(8, 1) # TODO

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.embedding(x) # (B, 12800, 1024)
        x = self.encoder1(x) # (B, 12800, 1024)
        print(x.shape)
        x = self.upsampling(x) # (B, 51200, 256)
        print(x.shape)
        x = self.encoder2(x)
        print(x.shape)
        x = self.upsampling(x)
        print(x.shape)
        x = self.encoder3(x)
        print(x.shape)
        exit()
        x = self.fc(x)
        x = torch.reshape(x, (-1, img_H, img_W, 1))
        return output
