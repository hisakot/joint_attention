import csv
import glob
import json
import os
import statistics

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, data_dir, img_height, img_width, transform=None, is_train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        self.mmpose = []
        self.targets = []
        self.img_paths = []
        self.gt_paths = []
        self.H = img_height
        self.W = img_width
        
        mmpose_paths = glob.glob(data_dir + "/mmpose/*.json")
        img_paths = glob.glob(data_dir + "/frames/*.png")
        gt_paths = glob.glob(data_dir + "/gt_heatmap/*.PNG")

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
        mmpose = self.mmpose[idx]
        frame_id = mmpose["frame_id"]
        instances = mmpose["instances"]
        # kptmap = np.zeros((self.H, self.W))
        kpts = [] # (human_num * keypoints)

        for instance in instances:
            keypoints = instance["keypoints"] # 133 human points
            scores = instance["keypoint_scores"]
            '''
            if sum(score >= 0.5 for score in scores) > 133 / 5:
                for kpt in keypoints:
                    gaze_x = kpt[0]
                    gaze_y = kpt[1]
                    kptmap[int(round(gaze_y)) - 1][int(round(gaze_x)) - 1] = 1
        kptmap = kptmap[:, :, np.newaxis]
        kptmap = (kptmap * 255).astype(np.uint8)
        kptmap = torch.tensor(kptmap, dtype=torch.int64)
        '''
            if sum(score >= 0.5 for score in scores) > 133 / 5:
                kpts.append(keypoints)
        kptmap = generate_pose_heatmap(self.H, self.W, kpts, sigma=3) # 1, H, W
# kptmap = cv2.resize(kptmap, (1920, 960))
        kptmap = kptmap.astype(np.float32)
        kptmap /= 255.
        kptmap = np.transpose(kptmap, (2, 0, 1)) # C, H, W

        img = cv2.imread(self.img_paths[idx]) # H, W, C
# img = cv2.resize(img, (1920, 960))
        img = img.astype(np.float32)
        img /= 255.
        img = np.transpose(img, (2, 0, 1)) # C, H, W

        inputs = {"kptmap" : torch.tensor(kptmap, dtype=torch.float32),
                  "img" : torch.tensor(img, dtype=torch.float32)}

        # labels
        targets = cv2.imread(self.gt_paths[idx])
        targets = cv2.resize(targets, (384, 192))
        targets = np.transpose(targets, (2, 0, 1)) # C, H, W

        '''
        if self.transform:
            data = self.transform(data)
        '''

        return inputs, torch.tensor(targets, dtype=torch.float32)

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

def generate_pose_heatmap(img_height, img_width, keypoints, sigma=3):
    human_num = len(keypoints)
    pose_heatmap = np.zeros((human_num, img_height, img_width), dtype=np.float32)
    heatmap = np.zeros((1, img_height, img_width), dtype=np.float32)

    for i, kpts in enumerate(keypoints):
        for kpt in kpts:
            x = int(kpt[0])
            y = int(kpt[1])
            pose_heatmap[i, y, x] = 255

    ksize = int(6 * sigma) | 1 # XOR (ksize needs to be odd)
    for i in range(human_num):
        pose_heatmap[i] = cv2.GaussianBlur(pose_heatmap[i], (ksize, ksize), sigma)
        heatmap += pose_heatmap[i]

    heatmap = np.max(heatmap, axis=0)
    heatmap = heatmap[:, :, np.newaxis]
    heatmap_cat = np.concatenate([heatmap, heatmap], 2)
    heatmap_cat = np.concatenate([heatmap, heatmap_cat], 2)

    return heatmap_cat
