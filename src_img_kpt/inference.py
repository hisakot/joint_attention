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
import torch.nn.functional as F
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
import PJAE_spatial
import PJAE_conv


def test(test_dataloader, model, loss_function, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            inp = data[0]
            for key, val in inp.items():
                if torch.is_tensor(val):
                    inp[key] = val.to(device)
            kptmap = inp["kptmap"]
            gazecone = inp["gazecone_map"]
            img = inp["img"]
            inputs = torch.cat([img, gazecone, kptmap], dim=1)

            targets = data[1].to(device)

            pred = model(inputs)

            if loss_function == "cos_similarity":
# pred_1vec = pred.view(pred.size(0), -1)
# targets = targets.view(targets.size(0), -1)
                cos_loss = F.cosine_similarity(pred, targets, dim=2)
                loss = (1 - cos_loss).mean()
            elif loss_function == "MSE":
                loss = nn.MSELoss(pred, targets)
            elif loss_function == "MAE":
                lossfunc = nn.L1Loss()
                loss = lossfunc(pred, targets)
            print(loss)

            pred = pred.to("cpu").detach().numpy().copy()
            pred = np.squeeze(pred, 0)
            pred = np.transpose(pred, (1, 2, 0))
            pred *= 255.
            pred = pred.astype(np.uint8)
            # pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
            pred = cv2.resize(pred, (960, 480))
            cv2.imwrite("data/test/pred/" + str(i).zfill(6) + ".png", pred)
            print("------------")

def main():

    parser = argparse.ArgumentParser(description="Process some integers")
    parser.add_argument("--model", required=True, help="Write model path")
    args = parser.parse_args()

    cfg = config.Config()
    img_height = cfg.img_height
    img_width = cfg.img_width

    '''
    model = vision_transformer.SwinUnet(img_height=img_height, img_width=img_width, in_chans=4)
    model = resnet.ResNet50(pretrained=False, in_ch=4)
    model = swin_transformer_v2.SwinTransformerV2(img_height=img_height, img_width=img_width,
                                                  in_chans=4, output_H=img_height, output_W=img_width)
    model = PJAE_spatiotemporal.ModelSpatioTemporal(in_ch=4)
    '''
    model = PJAE_conv.ModelSpatial(in_ch=5)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        print("---------- Use GPU ----------")
    else:
        print("---------- Use CPU ----------")
    # model.half().to(device)
    model.to(device)

    # loss_function = "MSE"
    # loss_function = "MAE"
    loss_function = "cos_similarity"

    checkpoint = torch.load(args.model)
    if torch.cuda.device_count() >= 1:
        model.load_state_dict(checkpoint["pjae_spatial_state_dict"], strict=False)
    else:
        from collections import OrderedDict
        state_dict = OrderedDict()
        for k, v in checkpoint["pjae_spatial_state_dict"].items():
            name = k[7:] # remove "module."
            state_dict[name] = v
        model.load_state_dict(state_dict)
    model.eval()

    test_data_dir = "data/test"
    test_data = dataset.Dataset(test_data_dir, img_height=img_height, img_width=img_width,
                                transform=None, is_train=False)
    test_dataloader = DataLoader(test_data, batch_size=1,
                                 shuffle=False, num_workers=1)
    test(test_dataloader, model, loss_function, device)

if __name__ == "__main__":
    main()
