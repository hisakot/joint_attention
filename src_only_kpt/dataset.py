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
    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        self.data = []
        self.targets = []
        
        mmpose_paths = glob.glob(data_dir + "/mmpose/*.json")
        gaze_paths = glob.glob(data_dir + "/gaze_ann/*.json")

        for file in mmpose_paths:
            instances = load_mmpose_json(file)
            self.data.extend(instances)
        for file in gaze_paths:
            gazes = load_gaze_ann_json(file)
            self.targets.extend(gazes)

    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, idx):
        # inputs
        inputs = []
        data = self.data[idx]
        frame_id = data["frame_id"]
        instances = data["instances"]
        for instance in instances:
            keypoints = instance["keypoints"]
            face_kpt = keypoints[23:91]
            scores = instance["keypoint_scores"]
            face_scores = scores[23:91]

            new_face_kpt = []
            if sum(score >= 0.5 for score in face_scores) > 68 / 5:
                p1, p2, yaw, pitch, roll = get_head_direction(face_kpt)
                pt1_x, pt1_y, pt2_x, pt2_y = get_face_rectangle(face_kpt, yaw, pitch)

                pt1_x = max(int(pt1_x), 0)
                pt1_y = max(int(pt1_y), 0)
                pt2_x = min(int(pt2_x), 3839)
                pt2_y = min(int(pt2_y), 1919)
                gz1_x = int(max(0, min(p1[0], 3839)))
                gz1_y = int(max(0, min(p1[1], 1919)))
                gz2_x = int(max(0, min(p2[0], 3839)))
                gz2_y = int(max(0, min(p2[1], 1919)))
                face_rec_gaze = [pt1_x, pt1_y, pt2_x, pt2_y, gz1_x, gz1_y, gz2_x, gz2_y]
                inputs.extend(face_rec_gaze)

        # labels
        labels = []
        targets = self.targets[idx]
        for annotation in targets["annotations"]:
            bbox = annotation["bbox"] # (x, y, w, h)
            x = (bbox[0] + bbox[2] / 2) / 3840
            y = (bbox[1] + bbox[3] / 2) / 1920
            labels.extend([x, y])
            break

        if self.transform:
            data = self.transform(data)

        return torch.tensor(inputs, dtype=torch.long), torch.tensor(labels, dtype=torch.float32)

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

