import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter

import config
import dataset
import train
import transformer
import swin_transformer
import swin_transformer_v2
import vision_transformer
import resnet
import PJAE_spatiotemporal


def test(test_dataloader, model, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            inputs = data[0]
            kptmap = inputs["kptmap"]
            gazemap = inputs["gazeline_map"]
            gazeconemap = inputs["gazecone_map"]
            saliencymap = inputs["saliency_map"]
            img = inputs["img"]
            targets = data[1]
            batch_size = len(data[1])

            '''
            img = img.to(device)
            kptmap = kptmap.to(device)
            '''
            concat_list = [img, gazeconemap]
            concat = torch.cat(concat_list, dim=1)
            concat = concat.to(device)
            pred = model(concat)
            '''
            img_pred = swin_t(img)
            kpt_pred = swin_t(kptmap)
            pred = fuse(img_pred, kpt_pred)
            '''
            pred = torch.clamp(pred, min=-1e3, max=1e3)

            pred = pred.to("cpu").detach().numpy().copy()
            pred = np.squeeze(pred, 0)
            pred = np.transpose(pred, (1, 2, 0))
            pred *= 255.
            pred = pred.astype(np.uint8)
            pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
            pred = cv2.resize(pred, (960, 480))
            cv2.imwrite("data/test/pred/" + str(i).zfill(6) + ".png", pred)


def main():

    parser = argparse.ArgumentParser(description="Process some integers")
    parser.add_argument("--model", required=True, help="Write model path")
    args = parser.parse_args()

    cfg = config.Config()
    img_height = cfg.img_height
    img_width = cfg.img_width

    '''
    model = swin_transformer_v2.SwinTransformerV2(img_height=img_height, img_width=img_width,
                                                  in_chans=6, output_H=img_height, output_W=img_width)
    model = resnet.ResNet50(pretrained=False, in_ch=6)
    model = PJAE_spatiotemporal.ModelSpatioTemporal(in_ch=3)
    '''
    model = vision_transformer.SwinUnet(img_height=img_height, img_width=img_width, in_chans=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        print("---------- Use GPU ----------")
    else:
        print("---------- Use CPU ----------")
    model.half().to(device)

    checkpoint = torch.load(args.model)
    if torch.cuda.device_count() >= 1:
        model.load_state_dict(checkpoint["pjae_state_dict"])
    else:
        from collections import OrderedDict
        state_dict = OrderedDict()
        for k, v in checkpoint["pjae_state_dict"].items():
            name = k[7:] # remove "module."
            state_dict[name] = v
        model.load_state_dict(state_dict)
    model.eval()

    test_data_dir = "data/test"
    test_data = dataset.Dataset(test_data_dir, img_height=img_height, img_width=img_width,
                                transform=None, is_train=False)
    test_dataloader = DataLoader(test_data, batch_size=1,
                                 shuffle=False, num_workers=1)
    test(test_dataloader, model, device)

if __name__ == "__main__":
    main()
