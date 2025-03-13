import csv
import glob
import json
import os
import statistics

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import config

class Dataset(Dataset):
    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        self.mmpose = []
        self.targets = []
        self.img_paths = []
        self.gt_paths = []
        cfg = config.Config()
        self.H = cfg.img_height
        self.W = cfg.img_width
        
        mmpose_paths = glob.glob(data_dir + "/mmpose/*.json")
        mmpose_paths.sort()
        img_paths = glob.glob(data_dir + "/frames/*/*.png")
        img_paths.sort()
        gt_paths = glob.glob(data_dir + "/gt_heatmap/*/*.png")
        gt_paths.sort()

        for file in mmpose_paths:
            instances = load_mmpose_json(file)
            self.mmpose.extend(instances)
        self.img_paths = img_paths
        self.gt_paths = img_paths

    def __len__(self):
        return len(self.mmpose)

    def __getitem__(self, idx):
        # inputs
        inputs = []
        '''
        mmpose = self.mmpose[idx]
        frame_id = mmpose["frame_id"]
        instances = mmpose["instances"]
        kptmap = np.zeros((self.H, self.W))

        for instance in instances:
            keypoints = instance["keypoints"]
            scores = instance["keypoint_scores"]

            if sum(score >= 0.5 for score in scores) > 133 / 5:
                for kpt in keypoints:
                    gaze_x = kpt[0]
                    gaze_y = kpt[1]
                    kptmap[int(round(gaze_y)) - 1][int(round(gaze_x)) - 1] = 1
        kptmap = kptmap[:, :, np.newaxis]
        kptmap = (kptmap * 255).astype(np.uint8)
        kptmap = torch.tensor(kptmap, dtype=torch.int64)
        inputs.extend(kptmap)
        '''
        img = cv2.imread(self.img_paths[idx]) # H, W, C
        img = cv2.resize(img, (self.W, self.H))
        img = img.astype(np.float32)
        img /= 255.
        img = np.transpose(img, (2, 0, 1)) # C, H, W
        img = torch.tensor(img, dtype=torch.float16)

        # labels
        labels = []
        targets = cv2.imread(self.gt_paths[idx])
        targets = targets.astype(np.float32)
        targets = cv2.resize(targets, None, fx=0.1, fy=0.1)
        targets /= 255.
        targets = np.transpose(targets, (2, 0, 1)) # C, H, W
        targets = torch.tensor(targets, dtype=torch.float16)

        '''
        if self.transform:
            data = self.transform(data)
        '''

        return img, targets

def load_mmpose_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
        instances = data["instance_info"]

        return instances

def load_gaze_ann_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
        targets = data["items"]

        return targets

def get_face_rectangle(face_kpt, yaw, pitch):
    x = [row[0] for row in face_kpt]
    y = [row[1] for row in face_kpt]
    center_x = statistics.mean(x)
    center_y = statistics.mean(y)
    width = max(x) - min(x)
    if yaw > -40 and yaw < 40:
        pt1_x = int(center_x - width)
        pt2_x = int(center_x + width)
    elif yaw <= -40:
        pt1_x = int(center_x - width / 2)
        pt2_x = int(center_x + 3 * width / 2)
    elif yaw >= 40:
        pt1_x = int(center_x - 3 * width / 2)
        pt2_x = int(center_x + width / 2)

    if pitch >= -15:
        pt1_y = int(center_y - width)
        pt2_y = int(center_y + width)
    else:
        pt1_y = int(center_y - 3 * width / 2)
        pt2_y = int(center_y + width / 2)

    return pt1_x, pt1_y, pt2_x, pt2_y

def get_head_direction(face_kpt):
    image_points = np.array([
        tuple(face_kpt[30]),
        tuple(face_kpt[21]),
        tuple(face_kpt[22]),
        tuple(face_kpt[39]),
        tuple(face_kpt[42]),
        tuple(face_kpt[31]),
        tuple(face_kpt[35]),
        tuple(face_kpt[48]),
        tuple(face_kpt[54]),
        tuple(face_kpt[57]),
        tuple(face_kpt[8]),], dtype='double')

    model_points = np.array([
        (0.0, 0.0, 0.0), # 30
        (-30.0, -125.0, -30.0), # 21
        (30.0,-125.0,-30.0), # 22
        (-60.0,-70.0,-60.0), # 39
        (60.0,-70.0,-60.0), # 42
        (-40.0,40.0,-50.0), # 31
        (40.0,40.0,-50.0), # 35
        (-70.0,130.0,-100.0), # 48
        (70.0,130.0,-100.0), # 54
        (0.0,158.0,-10.0), # 57
        (0.0,250.0,-50.0) # 8
        ])

    size = (1920, 3840, 3) # img.shape
    focal_length = size[1]
    center = (size[1] // 2, size[0] // 2) # center of face

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]], dtype='double')

    dist_coeffs = np.zeros((4, 1))

    (success, rotation_vector, translation_vector) = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE)
    (rotation_matrix, jacobian) = cv2.Rodrigues(rotation_vector)
    mat = np.hstack((rotation_matrix, translation_vector))

    (_, _, _, _, _, _, eulerAngles) = cv2.decomposeProjectionMatrix(mat)
    yaw = eulerAngles[1]
    pitch = eulerAngles[0]
    roll = eulerAngles[2]

    (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]),
                                              rotation_vector, translation_vector,
                                              camera_matrix, dist_coeffs)
    
    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    return p1, p2, yaw, pitch, roll

