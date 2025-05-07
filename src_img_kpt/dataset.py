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

class Dataset(Dataset):
    def __init__(self, data_dir, img_height, img_width, transform=None, is_train=True):
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
        gazecone_paths = glob.glob(data_dir + "/gazecone_close/*/*.png")
        gazecone_paths.sort()
        # gazecone_nch_paths = glob.glob(data_dir + "/gazecone_nch/*/*.npz")
        # gazecone_nch_paths.sort()
        kpt_paths = glob.glob(data_dir + "/kptmap/*/*.png")
        kpt_paths.sort()
        gt_paths = glob.glob(data_dir + "/gt_heatmap_1ch/*/*.png")
        gt_paths.sort()

        for file in mmpose_paths:
            instances = load_mmpose_json(file)
            self.mmpose.extend(instances)
        self.img_paths = img_paths
        self.gt_paths = gt_paths
        self.gazecone_paths = gazecone_paths
        # self.gazecone_nch_paths = gazecone_nch_paths
        self.kpt_paths = kpt_paths

    def __len__(self):
        return len(self.mmpose)

    def __getitem__(self, idx):
        # rotation anguler
        roll = random.uniform(0, 0) # FIXME if need, change angular range
        pitch = random.uniform(0, 0) # FIXME if need, change angular range
        yaw = random.uniform(0, 360)
        map_x, map_y = rotate_omni_img(self.H, self.W, roll, pitch, yaw)

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
            if sum(score >= 0.5 for score in scores) > 133 / 5:
                kpts.append(keypoints)
        # whole body keypoints
        '''
        kptmap = generate_pose_heatmap(self.H, self.W, kpts, sigma=3) # H, W, 1
        kptmap = cv2.remap(kptmap, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        kptmap = kptmap[:, :, np.newaxis]
        kptmap = kptmap.astype(np.float32)
        kptmap /= 255.
        kptmap = np.transpose(kptmap, (2, 0, 1)) # C, H, W
        '''

        # lined keypoints
        kptmap = cv2.imread(self.kpt_paths[idx], 0) # H, W
        kptmap = cv2.resize(kptmap, (self.W, self.H))
        kptmap = cv2.remap(kptmap, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        kptmap = kptmap.astype(np.float32)
        kptmap /= 255.
        kptmap = kptmap[np.newaxis, :, :] # 1, H, W


        # gaze_vector
        '''
        gaze_vector = []
        people_num = len(kpts)
        for kpt in kpts:
            face_kpt = kpt[23:91]
            p1, p2, yaw, pitch, roll = get_head_direction(face_kpt)
            # rotation
            loc_h = (map_x - p1[0])**2 + (map_y - p1[1])**2
            loc_g = (map_x - p2[0])**2 + (map_y - p2[1])**2
            hx, hy = np.unravel_index(np.argmin(loc_h), loc_h.shape)
            gx, gy = np.unravel_index(np.argmin(loc_g), loc_g.shape)
            hx /= 3840
            hy /= 1920
            gx /= 3840
            gy /= 1920
            gaze_vector.append([hx, hy, gx, gy]) 
        gaze_vector = torch.tensor(gaze_vector, dtype=torch.float32)
        '''

        # gaze line
        '''
        gazeline_map = np.zeros((1920, 3840, 3))
        for kpt in kpts:
            face_kpt = kpt[23:91]
            p1, p2, yaw, pitch, roll = get_head_direction(face_kpt)
            cv2.line(gazeline_map, p1, p2, (225, 225, 255), thickness=10)
        gazeline_map = cv2.resize(gazeline_map, (self.W, self.H))
        gazeline_map = cv2.remap(gazeline_map, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        gazeline_map = gazeline_map.astype(np.float32)
        gazeline_map /= 255
        gazeline_map = np.transpose(gazeline_map, (2, 0, 1)) # C, H, W
        gazeline_map = gazeline_map[0]
        gazeline_map = gazeline_map[np.newaxis, :, :]
        '''

        # gaze cone
        gazecone_map = cv2.imread(self.gazecone_paths[idx], 0) # H, W
        gazecone_map = cv2.resize(gazecone_map, (self.W, self.H))
        gazecone_map = cv2.remap(gazecone_map, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        gazecone_map = gazecone_map.astype(np.float32)
        gazecone_map /= 255.
        gazecone_map = gazecone_map[np.newaxis, :, :] # 1, H, W

        # gaze cone nch
        '''
        gazecone_nch = np.load(self.gazecone_nch_paths[idx])
        gazecone_nch = gazecone_nch['arr_0']
        height, width, people_num = gazecone_nch.shape
        gazecone_list = []
        for i in range(people_num):
            one_person_map = gazecone_nch[:, :, i]
            one_person_map = cv2.resize(one_person_map, (self.W, self.H))
            one_person_map = cv2.remap(one_person_map, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
            one_person_map = one_person_map[:, :, np.newaxis]
            gazecone_list.append(one_person_map)
        gazecone_nch_map = np.concatenate(gazecone_list, axis=2)
        gazecone_nch_map = gazecone_nch_map.astype(np.float32)
        gazecone_nch_map = np.transpose(gazecone_nch_map, (2, 0, 1)) # C, H, W
        # gazecone_nch_map /= 255.
        '''

        # saliency
        '''
        img = cv2.imread(self.img_paths[idx])
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        (success, saliency_map) = saliency.computeSaliency(img)
        saliency_map = cv2.resize(saliency_map, (self.W, self.H))
        saliency_map = cv2.remap(saliency_map, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        saliency_map = (saliency_map * 255).astype("uint8")
        saliency_map = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
        saliency_map = saliency_map.astype(np.float32)
        saliency_map /= 255.
        saliency_map = np.transpose(saliency_map, (2, 0, 1))
        '''

        # frame image
        img = cv2.imread(self.img_paths[idx], 0) # H, W, C (gray scale)
        img = cv2.resize(img, (self.W, self.H))
        img = cv2.remap(img, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        img = img[:, :, np.newaxis] # H, W, 1
        img = img.astype(np.float32)
        img /= 255.
        img = np.transpose(img, (2, 0, 1)) # C, H, W

        inputs = {"kptmap" : torch.tensor(kptmap, dtype=torch.float32),
                  # "gaze_vector" : torch.tensor(gaze_vector, dtype=torch.float32),
                  # "gazeline_map" : torch.tensor(gazeline_map, dtype=torch.float32),
                  "gazecone_map" : torch.tensor(gazecone_map, dtype=torch.float32),
                  # "gazecone_nch_map" : torch.tensor(gazecone_nch_map, dtype=torch.float32),
                  # "saliency_map" : torch.tensor(saliency_map, dtype=torch.float32),
                  "img" : torch.tensor(img, dtype=torch.float32)}

        # labels
        targets = cv2.imread(self.gt_paths[idx], 0) # Gray scale
        targets = cv2.resize(targets, (self.W, self.H))
        img = cv2.remap(targets, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        targets = targets[:, :, np.newaxis]
        targets = targets.astype(np.float32)
        targets /= 255.
        targets = np.transpose(targets, (2, 0, 1)) # C, H, W
        targets = torch.tensor(targets, dtype=torch.float32)

        '''
        if self.transform:
            data = self.transform(data)
        '''

        return inputs, targets

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

    (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 2000.0)]),
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
            if x >= 0 and x < img_width and y >= 0 and y < img_height:
                pose_heatmap[i, y, x] = 255

    ksize = int(6 * sigma) | 1 # XOR (ksize needs to be odd)
    for i in range(human_num):
        pose_heatmap[i] = cv2.GaussianBlur(pose_heatmap[i], (ksize, ksize), sigma)
        heatmap += pose_heatmap[i]

    heatmap = np.max(heatmap, axis=0)
    heatmap = heatmap[:, :, np.newaxis]

    return heatmap

def generate_gaze_cone(heatmap, p1, p2, sigma_angle, sigma_distance, fade_distance=False):
    height, width = heatmap.shape
    y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    norm = np.sqrt(dx**2 + dy**2)
    dx /= norm
    dy /= norm

    pixel_theta = np.arctan2(y_indices - p1[1], x_indices - p1[0]) # direction between each pixel and head center
    gaze_theta = np.arctan2(dy, dx) # gaze direction

    theta_diff = pixel_theta - gaze_theta
    theta_diff = np.arctan2(np.sin(theta_diff), np.cos(theta_diff))

    # gaze strength based on gauss
    angle_weight = np.exp(- (theta_diff ** 2) / (2 * sigma_angle ** 2))

    if fade_distance:
        distances = np.sqrt((x_indices - p1[0])**2 + (y_indices - p1[1])**2)
        distance_weight = np.exp(- (distances ** 2) / (2 * sigma_distance ** 2))
    else:
        distance_weight = 1

    heatmap += angle_weight * distance_weight

    return heatmap

def rotate_omni_img(H, W, roll, pitch, yaw):
    w = W // 2
    h = H // 2

    # make (H, W) grid
    theta = np.linspace(-np.pi, np.pi, W)
    phi = np.linspace(np.pi/2, -np.pi/2, H)
    theta, phi = np.meshgrid(theta, phi)

    # 3d coordinate
    x = np.cos(phi) * np.cos(theta)
    y = np.cos(phi) * np.sin(theta)
    z = np.sin(phi)

    # rolling
    roll = roll * np.pi / 180
    pitch = pitch * np.pi / 180
    yaw = yaw * np.pi / 180

    # 3d rollong array
    mtx1 = np.array([[1, 0, 0],
                     [0, np.cos(roll), np.sin(roll)],
                     [0, -np.sin(roll), np.cos(roll)]])
    mtx2 = np.array([[np.cos(pitch), 0, -np.sin(pitch)],
                     [0, 1, 0],
                     [np.sin(pitch), 0, np.cos(pitch)]])
    mtx3 = np.array([[np.cos(yaw), np.sin(yaw), 0],
                     [-np.sin(yaw), np.cos(yaw), 0],
                     [0, 0, 1]])

    # inner product of rolling array
    mtx4 = np.dot(mtx3, np.dot(mtx2, mtx1))

    # formula of coordinate
    xx = mtx4[0][0] * x + mtx4[0][1] * y + mtx4[0][2] * z
    yy = mtx4[1][0] * x + mtx4[1][1] * y + mtx4[1][2] * z
    zz = mtx4[2][0] * x + mtx4[2][1] * y + mtx4[2][2] * z

    # transport to latitude and longitude
    phi = np.arcsin(zz) / (np.pi / 2)
    theta = np.arctan2(yy, xx) / np.pi

    # origin is center of img
    X = theta * w
    Y = phi * h

    # origin is left upper
    map_x = X + w
    map_y = -Y + h

    return map_x, map_y
