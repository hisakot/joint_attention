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
import classifier

ROLL = ["NotHuman", "Surgeon", "Assistant", "Anesthesiology", "ScrubNurse", "CirculationNurse", "Visitor"]
COLOR = [(0, 0, 0), (0, 0, 255), (255, 0, 0), (0, 255, 0), (128, 0, 128), (0, 128, 128), (255, 255,255)]

def tensor_to_numpy(tensor2d):
    npy2d = tensor2d.to("cpu").detach().numpy().copy()
    # npy2d = np.squeeze(npy2d, 0)
    npy2d = np.transpose(npy2d, (1, 2, 0))
    npy2d *= 255
    npy2d = npy2d.astype(np.uint8)
    npy2d = cv2.resize(npy2d, (3840, 1920))
    # npy2d = cv2.applyColorMap(npy2d, cv2.COLORMAP_JET)
    return npy2d

def collate_function(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, targets

def test(test_dataloader, model, loss_function, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        img_num = 0
        for images, targets in test_dataloader:
            np_img = None
            for img, target in zip(images, targets):
                img = img.to(device)
                bboxes = target["bboxes"].to(device)
                labels = target["labels"].to(device)

                pred = model(img.unsqueeze(0), [bboxes])

                if loss_function[0] == "cos_similarity":
                    pred = pred.view(pred.size(0), -1)
                    labels = labels.view(labels.size(0), -1)
                    cos_loss = F.cosine_similarity(pred, labels)
                    loss = (1 - cos_loss).mean()
                elif loss_function[0] == "MSE":
                    lossfunc = nn.MSELoss()
                    loss = lossfunc(pred, labels)
                elif loss_function[0] == "CrossEntropyLoss":
                    labels = torch.argmax(labels, dim=1)
                    lossfunc = nn.CrossEntropyLoss()
                    loss = lossfunc(pred, labels)
                else:
                    print("Loss function is wrong")
                print(loss.item())

                if np_img is None:
                    np_img = tensor_to_numpy(img)
                for i, bbox in enumerate(bboxes):
                    bbox = [int(v) for v in bbox]
                    roll_num = torch.argmax(pred[i])
                    np_img = cv2.rectangle(np_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                           color=COLOR[roll_num], thickness=2)
                    np_img = cv2.putText(np_img, ROLL[roll_num]+" : "+str(pred[i][roll_num]),
                                         org=(bbox[0], bbox[1]), 
                                         fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1,
                                         color=COLOR[roll_num], thickness=1)
                cv2.imwrite("data/test/pred/roll_pred/" + str(img_num).zfill(6) + ".png", np_img)
                print("------------")
                img_num += 1

def main():

    parser = argparse.ArgumentParser(description="Process some integers")
    parser.add_argument("--model", required=True, help="Write model path")
    args = parser.parse_args()

    cfg = config.Config()
    img_height = cfg.img_height
    img_width = cfg.img_width

    model = classifier.ROIClassifier(num_classes=7)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 2:
        print("---------- Use GPUs ----------")
    else:
        print(f"---------- Use {device} ----------")
    # model.half().to(device)
    model.to(device)

    loss_function = ["CrossEntropyLoss"]
    # loss_function = ["MSE"]
    # loss_function = ["MAE"]
    # loss_function = ["cos_similarity"]

    checkpoint = torch.load(args.model)
    if torch.cuda.device_count() >= 1:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        from collections import OrderedDict
        state_dict = OrderedDict()
        for k, v in checkpoint["model_state_dict"].items():
            name = k[7:] # remove "module."
            state_dict[name] = v
        model.load_state_dict(state_dict)
    model.eval()

    test_data_dir = "data/test"
    test_data = dataset.Dataset(test_data_dir, img_height=img_height, img_width=img_width)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False,
                                 num_workers=1, pin_memory=True,
                                 collate_fn=collate_function)
    test(test_dataloader, model, loss_function, device)

if __name__ == "__main__":
    main()
