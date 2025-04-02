import csv
import json

import cv2
import numpy as np
from tqdm import tqdm

def generate_pose_heatmap(img_height, img_width, keypoints, sigma=3):
    human_num = len(keypoints)
    pose_heatmap = np.zeros((human_num, img_height, img_width), dtype=np.float32)
    heatmap = np.zeros((1, img_height, img_width), dtype=np.float32)

    for i, kpts in enumerate(keypoints):
        for kpt in kpts:
            x = int(kpt[0])
            y = int(kpt[1])
            pose_heatmap[i, y, x] = 255

    ksize = int(9 * sigma) | 1
    for i in range(human_num):
        pose_heatmap[i] = cv2.GaussianBlur(pose_heatmap[i], (ksize, ksize), sigma)
        heatmap += pose_heatmap[i]

    heatmap = np.max(heatmap, axis=0)
    heatmap = heatmap[:, :, np.newaxis]
    print("heatmap: ", heatmap.shape, heatmap.dtype)
    cat_heatmap = np.concatenate([heatmap, heatmap], 2)
    cat_heatmap = np.concatenate([cat_heatmap, heatmap], 2)
    cat_heatmap = cv2.resize(cat_heatmap, None, fx=0.25, fy=0.25)
    cat_heatmap = cat_heatmap.astype(np.float32)
    print("cat_heatmap: ", cat_heatmap.shape, cat_heatmap.dtype)

    cv2.imshow("heatmap", cat_heatmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("./kptmap.png", cat_heatmap)

    return heatmap

with open('data/train/mmpose/results_part009_35000_36000.json') as f:
    data = json.load(f)

    print("instance_info")
    print(len(data["instance_info"]))
    print(data["instance_info"][0].keys())
    print(data["instance_info"][0]["frame_id"])
    print(type(data["instance_info"][0]["instances"]))

    print("-------")

    instances = data["instance_info"][0]["instances"]
    print("human number of one image: ", len(instances))
    keypoints = []
    for instance in instances:
        keypoint = instance["keypoints"]
        scores = instance["keypoint_scores"]
        if sum(score >- 0.5 for score in scores) > 133 / 5:
            keypoints.append(keypoint)
    kptmap = generate_pose_heatmap(1920, 3840, keypoints, sigma=3)
