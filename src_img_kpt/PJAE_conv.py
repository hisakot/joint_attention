import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import cv2
import numpy as np

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

class VariableLengthVectorToImage(nn.Module):
    def __init__(self, input_dim, image_height=320, image_width=640, patch_size=8, embed_dim=256, num_heads=4, num_layers=6):
        super().__init__()

        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1024, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.image_height = image_height
        self.image_width = image_width
        self.patch_size = patch_size
        self.num_patches = (image_height // patch_size) * (image_width // patch_size)
        self.project = nn.Linear(embed_dim, 1 * patch_size * patch_size) # TODO changed 3 -> 1
        self.unpatchfy = PatchToImage(patch_size, image_height, image_width, channels=1)

    def forward(self, x): # x: [B, L, input_dim]
        B, L, _ = x.shape
        x = self.embedding(x) + self.pos_encoding[:L] # [B, L, D]
        x = x.permute(1, 0, 2) # [L, B, D] -> Transformer expects time first
        x = self.transformer(x)
        x = x.permute(1, 0, 2) # [B, L, D]

        pooled = x.mean(dim=1)
        expanded = pooled.unsqueeze(1).repeat(1, self.num_patches, 1) # [B, N, D]

        patches = self.project(expanded) # [B, N, patch_dim]
        img = self.unpatchfy(patches)

        return img

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ModelSpatial(nn.Module):
    def __init__(self, block=Bottleneck, layers_scene=[3, 4, 6, 3, 2], layers_face=[3, 4, 6, 3, 2], in_ch=3):
        self.inplanes_scene = 64
        self.inplanes_face = 64
        super(ModelSpatial, self).__init__()

        # TODO added
        self.fixed_in_ch = in_ch
        self.input_convs = nn.ModuleDict()
        self.sigmoid = nn.Sigmoid()

        # common
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # scene pathway
        self.conv1_scene = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_scene = nn.BatchNorm2d(64)
        self.layer1_scene = self._make_layer_scene(block, 64, layers_scene[0])
        self.layer2_scene = self._make_layer_scene(block, 128, layers_scene[1], stride=2)
        self.layer3_scene = self._make_layer_scene(block, 256, layers_scene[2], stride=2)
        self.layer4_scene = self._make_layer_scene(block, 512, layers_scene[3], stride=2)
        self.layer5_scene = self._make_layer_scene(block, 256, layers_scene[4], stride=1) # additional to resnet50

        # face pathway
        self.conv1_face = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_face = nn.BatchNorm2d(64)
        self.layer1_face = self._make_layer_face(block, 64, layers_face[0])
        self.layer2_face = self._make_layer_face(block, 128, layers_face[1], stride=2)
        self.layer3_face = self._make_layer_face(block, 256, layers_face[2], stride=2)
        self.layer4_face = self._make_layer_face(block, 512, layers_face[3], stride=2)
        self.layer5_face = self._make_layer_face(block, 256, layers_face[4], stride=1) # additional to resnet50

        # attention
        self.attn = nn.Linear(1808, 1*7*7)

        # encoding for saliency
        self.compress_conv1 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, bias=False) # 2048->1024
        self.compress_bn1 = nn.BatchNorm2d(1024)
        self.compress_conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2 = nn.BatchNorm2d(512)

        # encoding for in/out
        self.compress_conv1_inout = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False) # 2048->1024
        self.compress_bn1_inout = nn.BatchNorm2d(512)
        self.compress_conv2_inout = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2_inout = nn.BatchNorm2d(1)
        self.fc_inout = nn.Linear(300, 1) # TODO (49, 1) <- img size 224, 224

        # decoding
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        # self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv_bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        # self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv_bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 1, kernel_size=3, stride=2)
        # self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv_bn3 = nn.BatchNorm2d(1)
        # self.deconv_bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=1, stride=1)
        # self.conv4 = nn.Conv2d(64, 1, kernel_size=1, stride=1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer_scene(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_scene != planes * block.expansion:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes_scene, planes * block.expansion,
                             kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                    )
        layers = []
        layers.append(block(self.inplanes_scene, planes, stride, downsample))
        self.inplanes_scene = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_scene, planes))

        return nn.Sequential(*layers)

    def _make_layer_face(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_face != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_face, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_face, planes, stride, downsample))
        self.inplanes_face = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_face, planes))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        batch_size, inp_ch, inp_height, inp_width = inputs.shape
        resousion_height, resousion_width = 224, 448 
        frame_num = 1
        people_num = 1

        inputs = inputs.view(batch_size*frame_num, inp_ch, inp_height, inp_width)
        inputs = F.interpolate(inputs, (resousion_height, resousion_width), mode='bilinear')

        im = self.conv1_scene(inputs)
        im = self.bn1_scene(im)
        im = self.relu(im)
        im = self.maxpool(im)
        im = self.layer1_scene(im)
        im = self.layer2_scene(im)
        im = self.layer3_scene(im)
        im = self.layer4_scene(im)
        scene_feat = self.layer5_scene(im) # (B, 1024, 7, 14)

        encoding = self.compress_conv1(scene_feat)
        encoding = self.compress_bn1(encoding)
        encoding = self.relu(encoding)
        encoding = self.compress_conv2(encoding)
        encoding = self.compress_bn2(encoding)
        encoding = self.relu(encoding) # (B, 512, 7, 14)

        x = self.deconv1(encoding)
        x = self.deconv_bn1(x)
        x = self.relu(x) # (B, 256, 15, 29)
        x = self.deconv2(x)
        x = self.deconv_bn2(x)
        x = self.relu(x) # (B, 128, 31, 59)
        x = self.deconv3(x)
        x = self.deconv_bn3(x)
        x = self.relu(x) # (B, 1, 63, 119)
        x = self.conv4(x) # (B, 1, 63, 119)
        x = self.sigmoid(x) # (B, 1, 63, 119)

        # return deconv, inout_val, hx
        x = F.interpolate(x, (320, 640), mode='bilinear', align_corners=False)
        output = x

        return output


    def calc_loss(self, inp, out, cfg):
        att_inside_flag = inp['att_inside_flag']
        encoding_inout = out['encoding_inout']
        inout = 1 / (1 + torch.exp(-encoding_inout))
        inout = (1 - inout) * 255

        loss_set = {}
        return loss_set

