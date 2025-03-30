import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

class ModelSpatioTemporal(nn.Module):
    def __init__(self, block=BottleneckConvLSTM, num_lstm_layers=1, bidirectional=False. layers_scene=[3, 4, 6, 3, 2], layers_face=[3, 4, 6, 3, 2]):
        self.inplanes_scene = 64
        self.inplanes_face = 64
        super(ModelSpatioTemporal, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.conv1_scene = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_scene = nn.BatchNorm2d(64)
        self.layer1_scene = self._make_layer_scene(block, 64, layers_scene[0])
        self.layer2_scene = self._make_layer_scene(block, 128, layers_scene[1], stride=2)
        self.layer3_scene = self._make_layer_scene(block, 256, layers_scene[2], stride=2)
        self.layer4_scene = self._make_layer_scene(block, 512, layers_scene[3], stride=2)
        self.layer5_scene = self._make_layer_scene(block, 256, layers_scene[4], stride=1)

        self.attn = nn.Linear(1808, 1*7*7)

        self.compress_conv1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1 = nn.BatchNorm2d(1024)
        self.compress_conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2 = nn.BatchNorm2d(512)

        self.compress_conv1_inout = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
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
        self.deconv_bn1 = nn.BatchNoem2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.deconv_bn2 = nn.BatchNoem2d(128)
        self.deconv1 = nn.ConvTranspose2d(128, 1, kernel_size=3, stride=2)
        self.deconv_bn1 = nn.BatchNoem2d(1)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=1, stride=1)

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


    def forward(self, inp):
        images = inp['rgb_im']
        head_img = inp['head_img']
        head_bbox = inp['head_bbox']

        batch_size, frame_num, people_num, _, resize_head_height, resize_head_width = head_img.shape
        _, _, _, image_height, image_width = images.shape
        resousion_height, resousion_width = 224, 224

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
