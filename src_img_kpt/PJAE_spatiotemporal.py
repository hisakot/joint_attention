import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
from pytorch_convolutional_rnn import convolutional_rnn

class BottleneckConvLSTM(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckConvLSTM, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.bn_ds = nn.BatchNorm2d(planes * self.expansion)

        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # RW edit: handles batch_size==1
        if out.shape[0] > 1:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # RW edit: handles batch_size==1
        if out.shape[0] > 1:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # RW edit: handles batch_size==1
        if out.shape[0] > 1:
            out = self.bn3(out)

        if self.downsample is not None:
            # RW edit: handles batch_size==1
            if out.shape[0] > 1:
                residual = self.downsample(x)
                residual = self.bn_ds(residual)
            else:
                residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class ModelSpatioTemporal(nn.Module):
    def __init__(self, block=BottleneckConvLSTM, num_lstm_layers=1, bidirectional=False, layers_scene=[3, 4, 6, 3, 2], layers_face=[3, 4, 6, 3, 2]):
        self.inplanes_scene = 64
        self.inplanes_face = 64
        super(ModelSpatioTemporal, self).__init__()

        # common
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # scene pathway
        self.conv1_scene = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_scene = nn.BatchNorm2d(64)
        self.layer1_scene = self._make_layer_scene(block, 64, layers_scene[0])
        self.layer2_scene = self._make_layer_scene(block, 128, layers_scene[1], stride=2)
        self.layer3_scene = self._make_layer_scene(block, 256, layers_scene[2], stride=2)
        self.layer4_scene = self._make_layer_scene(block, 512, layers_scene[3], stride=2)
        self.layer5_scene = self._make_layer_scene(block, 256, layers_scene[4], stride=1) # additional to resnet50

        # face pathway
        self.conv1_face = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
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

        self.convlstm_scene = convolutional_rnn.Conv2dLSTM(in_channels=512, 
                                                           out_channels=512,
                                                           kernel_size=3,
                                                           num_layers=num_lstm_layers,
                                                           bidirectional=bidirectional,
                                                           batch_first=True,
                                                           stride=1,
                                                           dropout=0.5)

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
        images = inp
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

        # version3
        encoding = encoding.view(batch_size, frame_num, people_num, 512, 7, 7)
        encoding = encoding.permute(1, 0, 2, 3, 4, 5)
        hx=None
        y_list = []
        for t in range(frame_num):
            encoding_t = encoding[t].view(batch_size, people_num, 512, 7, 7)
            y, hx = self.convlstm_scene(encoding_t, hx=hx)
            y_list.append(y)
        deconv = torch.stack(y_list, dim=0)
        deconv = deconv.view(frame_num, batch_size, people_num, 512, 7, 7)
        deconv = deconv.permute(1, 0, 2, 3, 4, 5)
        deconv = deconv.view(batch_size*frame_num*people_num, 512, 7, 7)

        inout_val = encoding_inout.view(-1, 49)
        inout_val = self.fc_inout(inout_val)

        deconv = self.deconv1(deconv)
        if encoding.shape[0] > 1:
            deconv = self.deconv_bn1(deconv)
        deconv = self.relu(deconv)
        deconv = self.deconv2(deconv)
        if encoding.shape[0] > 1:
            deconv = self.deconv_bn2(deconv)
        deconv = self.relu(deconv)
        deconv = self.deconv3(deconv)
        if encoding.shape[0] > 1:
            deconv = self.deconv_bn3(deconv)
        deconv = self.relu(deconv)
        deconv = self.conv4(deconv)

        raw_hm = deconv * 255
        inout = 1 / (1 + torch.exp(-inout_val))
        inout = (1 - inout) * 255
        x = raw_hm - inout[:, :, None, None]
        x = x / 255

        # pack output data
        out = {}
        out['encoding_inout'] = encoding_inout
        out['person_scene_attention_heatmap'] = x.view(batch_size, frame_num, people_num, 63, 63) # 64->63

        # return deconv, inout_val, hx
        '''
        return out
        '''
        x = F.interpolate(x, (320, 640), mode='bilinear')
        return x


    def calc_loss(self, inp, out, cfg):
        att_inside_flag = inp['att_inside_flag']
        encoding_inout = out['encoding_inout']
        inout = 1 / (1 + torch.exp(-encoding_inout))
        inout = (1 - inout) * 255

        loss_set = {}
        return loss_set

