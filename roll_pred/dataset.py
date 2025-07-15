import csv
import glob
import json
import os
import random
import statistics

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import config

ROLL = ["NotHuman", "Surgeon", "Assistant", "Anesthesiology", "ScrubNurse", "CirculationNurse", "Visitor"]

class Dataset(Dataset):
    def __init__(self, data_dir, img_height, img_width, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.mmpose = []
        self.img_paths = []
        cfg = config.Config()
        self.H = cfg.img_height
        self.W = cfg.img_width
        
        mmpose_paths = glob.glob(data_dir + "/roll_ann/mmpose/*.json")
        mmpose_paths.sort()
        img_paths = glob.glob(data_dir + "/roll_ann/frames/*/*.png")
        img_paths.sort()

        for file in mmpose_paths:
            instances = load_mmpose_json(file)
            self.mmpose.extend(instances)
        self.img_paths = img_paths

    def __len__(self):
        return len(self.mmpose)

    def __getitem__(self, idx):
        # inputs
        inputs = []
        mmpose = self.mmpose[idx]
        frame_id = mmpose["frame_id"]
        instances = mmpose["instances"]

        bboxes = []
        rolls = []
        for instance in instances:
            bbox = instance["bbox"][0]
            bbox = [int(v) for v in bbox]
            bboxes.append(bbox)
            roll = [0] * len(ROLL)
            roll[ROLL.index(instance["roll"])] += 1
            rolls.append(roll)

        # frame image
        img = cv2.imread(self.img_paths[idx], 1) # H, W, C (gray scale-> 0)
        # img = img[:, :, np.newaxis] # H, W, 1 if Gray scale
        img = img.astype(np.float32)
        img /= 255.
        img = np.transpose(img, (2, 0, 1)) # C, H, W
        img = torch.tensor(img, dtype=torch.float32)

        # labels
        targets = {"bboxes" : torch.tensor(bboxes, dtype=torch.float32),
                   "labels" : torch.tensor(rolls, dtype= torch.float32),
                   }

        return img, targets

def load_mmpose_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
        instances = data["instance_info"]

        return instances

