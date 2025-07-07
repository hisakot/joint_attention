import argparse
import glob
import json
import os
import time

import cv2
import matplotlib.pyplot as plt
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
'''
import transformer
import swin_transformer
import swin_transformer_v2
import vision_transformer
import resnet
import PJAE_spatiotemporal
import PJAE_spatial
import vis_transformer
'''
import PJAE_conv
import cnn_transformer

def tensor_to_numpy(tensor2d):
    npy2d = tensor2d.to("cpu").detach().numpy().copy()
    npy2d = np.squeeze(npy2d, 0)
    npy2d = np.transpose(npy2d, (1, 2, 0))
    npy2d *= 255
    npy2d = npy2d.astype(np.uint8)
    npy2d = cv2.resize(npy2d, (960, 480))
    # npy2d = cv2.applyColorMap(npy2d, cv2.COLORMAP_JET)
    return npy2d

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
            pred_sigmoid = torch.sigmoid(pred)
            plt.imshow(pred_sigmoid[0, 0].cpu().numpy(), cmap='jet')
            plt.clorbar()
            plt.title("predicted heatmap")
            plt.show()
            exit()

            if loss_function[0] == "cos_similarity":
                pred_1vec = pred.view(pred.size(0), -1)
                targets_1vec = targets.view(targets.size(0), -1)
                cos_loss = F.cosine_similarity(pred_1vec, targets_1vec)
                loss = (1 - cos_loss).mean()
            elif loss_function[0] == "MSE":
                lossfunc = nn.MSELoss()
                loss = lossfunc(pred, targets)
            elif loss_function[0] == "MAE":
                lossfunc = nn.L1Loss()
                loss = lossfunc(pred, targets)
            elif loss_function[0] == "cos_MSE":
                alpha = loss_function[1]
                pred = pred.view(pred.size(0), -1)
                targets = targets.view(targets.size(0), -1)
                cos_loss = 1 - F.cosine_similarity(pred, targets).mean()
                mse_loss = F.mse_loss(pred, targets)
                loss = alpha * cos_loss + (1 - alpha) * mse_loss
            else:
                print("Loss function is wrong")
            print(loss)

            np_pred = tensor_to_numpy(pred)
            np_pred = np_pred[:, :, np.newaxis]
            cv2.imwrite("data/test/pred/result1/" + str(i).zfill(6) + ".png", np_pred)

            np_img = tensor_to_numpy(img)
            np_target = tensor_to_numpy(targets)
            np_target = np_target[:, :, np.newaxis]
            zero = np.zeros((480, 960, 1), dtype=np.uint8)
            result = np.concatenate([np_pred, zero, np_target], axis=2)
            result_img = cv2.addWeighted(np_img, 0.7, result, 1, 0)
            cv2.imwrite("data/test/pred/gaze_mult_allaround_augmentation_0/" + str(i).zfill(6) + ".png", result_img)
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
    model = swin_transformer_v2.SwinTransformerV2(img_height=img_height, img_width=img_width,
                                                  in_chans=5, output_H=img_height, output_W=img_width)
    model = vis_transformer.VisionTransformer(in_channels=5, patch_size=4, emb_size=64,
                                              img_H=img_height, img_W=img_width, num_layers=2,
                                              num_heads=2, forward_expansion=4, num_classes=128)
    '''
    model = PJAE_conv.ModelSpatial(in_ch=5)
    '''
    model = cnn_transformer.CNNTransformer2Heatmap(in_channels=5,
                                                   img_size=(img_height, img_width),
                                                   output_size=(img_height, img_width))
    '''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        print("---------- Use GPU ----------")
    else:
        print("---------- Use CPU ----------")
    # model.half().to(device)
    model.to(device)

    # loss_function = ["MSE"]
    # loss_function = ["MAE"]
    loss_function = ["cos_similarity"]
    # loss_function = ["cos_MSE", 0.5]

    checkpoint = torch.load(args.model)
    if torch.cuda.device_count() >= 1:
        model.load_state_dict(checkpoint["cnn_transformer_state_dict"], strict=False)
    else:
        from collections import OrderedDict
        state_dict = OrderedDict()
        for k, v in checkpoint["cnn_transformer_state_dict"].items():
            name = k[7:] # remove "module."
            state_dict[name] = v
        model.load_state_dict(state_dict)
    model.eval()

    test_data_dir = "data/test"
    test_data = dataset.Dataset(test_data_dir, img_height=img_height, img_width=img_width,
                                transform=None, is_train=False, inf_rotate=0)
    test_dataloader = DataLoader(test_data, batch_size=1,
                                 shuffle=False, num_workers=1)
    test(test_dataloader, model, loss_function, device)

if __name__ == "__main__":
    main()
