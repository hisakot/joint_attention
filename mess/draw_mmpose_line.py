import glob
import json
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

def generate_pose_heatmap(img_height, img_width, keypoints, confs, links, sigma=3):
    heatmap = np.zeros((img_height, img_width, 1), dtype=np.float32)

    for i, kpts in enumerate(keypoints):
        conf = confs[i]
        for j, kpt in enumerate(kpts):
            x = int(kpt[0])
            y = int(kpt[1])
            c = conf[j]
            if c < 0.6:
                continue
            if x >= 0 and x < img_width and y >= 0 and y < img_height:
                cv2.circle(heatmap, (x, y), radius=6, color=255, thickness=-1)
        for link in links:
            c = conf[link[0]] + conf[link[1]]
            if c < 1.2:
                continue
            pt1 = kpts[link[0]]
            pt2 = kpts[link[1]]
            cv2.line(heatmap, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), 255, thickness=3)
    heatmap = np.max(heatmap, axis=2)
    heatmap = heatmap[:, :, np.newaxis]

    return heatmap

def load_mmpose_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
        instances = data["instance_info"]

        return instances

def load_mmpose_links(mmpose_path):
    with open(mmpose_path) as f:
        data = json.load(f)
        meta_info = data["meta_info"]
        links = meta_info["skeleton_links"]
        return links

H = 960
W = 1920
data_dir = "data/Fig"
mmpose_paths = glob.glob(data_dir + "/mmpose/*.json")
mmpose_paths.sort()
mmposes = []
for file in mmpose_paths:
    instances = load_mmpose_json(file)
    mmposes.extend(instances)

i = -1
for mmpose in tqdm(mmposes):
    frame_id = mmpose["frame_id"]
    instances = mmpose["instances"]
    links = load_mmpose_links(mmpose_paths[0]) # list of skelton links
    kpts = [] # (human_num * keypoints)
    confs = []

    for instance in instances:
        keypoints = instance["keypoints"] # 133 human points
        scores = instance["keypoint_scores"]
        if sum(score >= 0.5 for score in scores) > 133 / 5:
            kpts.append(keypoints)
            confs.append(scores)

    # whole body keypoints
    kptmap = generate_pose_heatmap(H, W, kpts, confs, links, sigma=3) # H, W, 1
    if frame_id == 1:
        i += 1
    video_dir = os.path.splitext(os.path.basename(mmpose_paths[i]))[0]
    save_dir = os.path.join(data_dir, "kptmap/", video_dir)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    cv2.imwrite(os.path.join(save_dir, str(frame_id).zfill(6)) + ".png", kptmap)
    '''
    kptmap = kptmap[:, :, np.newaxis]
    kptmap = kptmap.astype(np.float32)
    kptmap /= 255.
    kptmap = np.transpose(kptmap, (2, 0, 1)) # C, H, W
    '''

