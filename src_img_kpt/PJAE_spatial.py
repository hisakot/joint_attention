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

class Fusion(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(Fusion, self).__init__()

        self.fusion_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        concat = torch.cat([x, y], dim=1)

        fused = self.fusion_layer(concat)
        output = self.sigmoid(fused)

        return output

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

        self.fusion = Fusion() # TODO added

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
        self.fc_inout = nn.Linear(49, 1)

        # decoding
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.deconv_bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.deconv_bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 1, kernel_size=3, stride=2)
        self.deconv_bn3 = nn.BatchNorm2d(1)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=1, stride=1)

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

    def forward(self, inp):
        '''
        images = inp['rgb_im']
        head_img = inp['head_img']
        head_bbox = inp['head_bbox']

        batch_size, frame_num, people_num, _, resize_head_height, resize_head_width = head_img.shape
        _, _, _, image_height, image_width = images.shape
        resousion_height, resousion_width = 224, 224
        '''
        image = inp["img"]
        gazecone = inp["gazecone_map"]
        images = torch.cat([image, gazecone], dim=1)
        batch_size, img_ch, image_height, image_width = images.shape
        resousion_height, resousion_width = 224, 224
        frame_num = 1
        people_num = 1

        '''
        head_bbox_flat = head_bbox.view(batch_size*frame_num*poeplw_num, 4)
        head = torch.zeros(batch_size*frame_num*people_num, 1, resousion_height, resousion_width, device=images.device)
        for head_idx, in range(batch_size* frame_num*pople_num):
            x_min, y_min, x_max, y_max = head_bbox_flat[head_idx, :]
            x_min, x_max = x_min*resousion_width, x_max*resousion_width
            y_min, y_max = y_min*resousion_height, y_max*resousion_height_
            x_min = x_min.long()
            x_max = x_max.long()
            y_min = y_min.long()
            y_max = y_max.long()
            head[head_idx, 0, y_min:y_max, x_min:x_max] = 1

        face = head_img.view(batch_size*frame_num*people_num, 3, resize_head_height, resize_head_width)
        face = F.interpolate(face, (resousion_height, resousion_width), mode='bilinear')
        '''
        images = images.view(batch_size*frame_num, img_ch, image_height, image_width)
        images = F.interpolate(images, (resousion_height, resousion_width), mode='bilinear')
        images = images.repeat(people_num, 1, 1, 1)

        '''
        face = self.conv1_face(face)
        face = self.bn1_face(face)
        face = self.relu(face)
        face = self.maxpool(face)
        face = self.layer1_face(face)
        face = self.layer2_face(face)
        face = self.layer3_face(face)
        face = self.layer4_face(face)
        face_feat = self.layer5_face(face)

        # reduce head channnel seze by max pooling: (N, 1, 224, 224) -> (N, 1, 28, 28)
        head_reduced = self.maxpool(self.maxpool(self.maxpool(head))).view(-1, 784)
        # reduce head channnel seze by avg pooling: (N, 1024, 7, 7) -> (N, 1024, 1, 1)
        face_feat_reduced = self.avgpool(face_feat).view(-1, 1024)
        # get and reshape attention weights such that it can be multiplied with scene feature map
        attn_weights = self.attn(torch.cat((head_reduced, face_feat_reduced), 1))
        attn_weights = attn_weights.view(-1, 1, 49)
        attn_weights = F.softmax(attn_weights, dim=2) # soft attention weights single-channel
        attn_weights = attn_weights.view(-1, 1, 7, 7)

        im = torch.cat((images, head), dim=1)
        '''
        im = self.conv1_scene(images)
        im = self.bn1_scene(im)
        im = self.relu(im)
        im = self.maxpool(im)
        im = self.layer1_scene(im)
        im = self.layer2_scene(im)
        im = self.layer3_scene(im)
        im = self.layer4_scene(im)
        scene_feat = self.layer5_scene(im)
        '''
        attn_applied_scene_feat = torch.mul(attn_weights, scene_feat) # (N, 1, 7, 7) # applying attention weights on scene feat

        scene_face_feat = torch.cat((attn_applied_scene_feat, face_feat), 1)
        '''

        # scene + face feat -> in/out
        '''
        encoding_inout = self.compress_conv1_inout(scene_face_feat)
        '''
        encoding_inout = self.compress_conv1_inout(scene_feat)
        encoding_inout = self.compress_bn1_inout(encoding_inout)
        encoding_inout = self.relu(encoding_inout)
        encoding_inout = self.compress_conv2_inout(encoding_inout)
        encoding_inout = self.compress_bn2_inout(encoding_inout)
        encoding_inout = self.relu(encoding_inout)
        encoding_inout = encoding_inout.view(-1, 49)
        encoding_inout = self.fc_inout(encoding_inout)

        # scene + face feat -> encoding -> decoding
        '''
        encoding = self.compress_conv1(scene_face_feat)
        '''
        encoding = self.compress_conv1(scene_feat)
        encoding = self.compress_bn1(encoding)
        encoding = self.relu(encoding)
        encoding = self.compress_conv2(encoding)
        encoding = self.compress_bn2(encoding)
        encoding = self.relu(encoding)

        x = self.deconv1(encoding)
        x = self.deconv_bn1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.deconv_bn2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.deconv_bn3(x)
        x = self.relu(x)
        x = self.conv4(x)

        raw_hm = x * 255
        inout = 1 / (1 + torch.exp(-encoding_inout))
        inout = (1 - inout) * 255
        x = raw_hm - inout[:, :, None, None]
        x = x / 255

        # pack output data
        '''
        out = {}
        out['encoding_inout'] = encoding_inout
        out['person_scene_attention_heatmap'] = x.view(batch_size, frame_num, people_num, 63, 63) # 64->63
        '''

        # return deconv, inout_val, hx
        '''
        return out
        '''
        x = F.interpolate(x, (320, 640), mode='bilinear')

        # gazevector
        gaze_vector = inp["gaze_vector"]
        B, L, D = gaze_vector.shape
        device = gaze_vector.device
        # gaze_vector_transformer = VariableLengthVectorToImage(input_dim=D).half().to(device) # TODO half ver
        gaze_vector_transformer = VariableLengthVectorToImage(input_dim=D).to(device)
        y = gaze_vector_transformer(gaze_vector)

        # fuse 
        output = self.fusion(x, y)

        return output


    def calc_loss(self, inp, out, cfg):
        att_inside_flag = inp['att_inside_flag']
        encoding_inout = out['encoding_inout']
        inout = 1 / (1 + torch.exp(-encoding_inout))
        inout = (1 - inout) * 255

        loss_set = {}
        return loss_set

