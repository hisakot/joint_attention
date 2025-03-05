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

import dataset
import train
import transformer
import swin_transformer
import swin_transformer_v2


def test(test_dataloader, model, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for i, (data, mask, targets, length) in enumerate(test_dataloader):
            img = data.to(device)
            pred = model(img)
            pred = pred.to("cpu").detach().numpy().copy()
            pred = np.squeeze(pred, 0)
            pred = np.transpose(pred, (1, 2, 0))
            pred *= 255.
            pred = pred.astype(np.uint8)
            pred = cv2.resize(pred, (3840, 1920))
            cv2.imwrite("data/test/pred/" + str(i).zfill(6) + ".png", pred)
            exit()

    return pred

def main():

    parser = argparse.ArgumentParser(description="Process some integers")
    parser.add_argument("--model", required=True, help="Write model path")
    args = parser.parse_args()

    img_height = 1920
    img_width = 3840

    model = swin_transformer_v2.SwinTransformerV2(img_height=img_height, img_width=img_width,
                                                  output_img_size=192*384)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        print("---------- Use GPU ----------")
        model = nn.DataParallel(model)
    else:
        print("---------- Use CPU ----------")
    model.half().to(device)

    checkpoint = torch.load(args.model)
    if torch.cuda.device_count() >= 1:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        from collections import OrderedDict
        state_dict = OrderedDict()
        for k, v in checkpoint["model_state_dict"].items():
            name = k[7:] # remove "module."
            state_dict[name] = v
        model.load_state_dict(state_dict)
    model.eval()

    test_data_dir = "data/val"
    test_data = dataset.Dataset(test_data_dir, transform=None, is_train=False)
    test_dataloader = DataLoader(test_data, batch_size=1,
                             collate_fn=train.collate_fn, num_workers=1) # FIXME collate_fn -> common.py or delete

    test(test_dataloader, model, device)

if __name__ == "__main__":
    main()
