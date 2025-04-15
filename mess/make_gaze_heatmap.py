import glob
import json
import os

import cv2
import numpy as np
from scipy import ndimage

def load_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
        targets = data["items"]

        return targets

W = 3840
H = 1920

gaze_ann_paths = glob.glob("data/train/gaze_ann/*.json")
for gaze_ann_path in gaze_ann_paths:
    data = load_json(gaze_ann_path)

    video_name = os.path.splitext(os.path.basename(gaze_ann_path))[0]
    save_dir = "data/train/gt_heatmap_1ch/" + video_name
    if os.path.exists(save_dir):
        print(save_dir + " is exits")
        continue
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for i, one_frame_ann in enumerate(data):
        ann_in_img = one_frame_ann["annotations"]
        gazemap = np.zeros((H, W))
        for ann in ann_in_img:
            bbox = ann["bbox"] # (x, y, w, h)
            x = bbox[0] + bbox[2] / 2 # (x + w/2) -> normalize
            y = bbox[1] + bbox[3] / 2 # (h + h/2) -> normalize

            gaze_point = np.array([int(x), int(y)])
            if gaze_point[0] == 0 and gaze_point[1] == 0:
                gaze_point = np.array([1, 1]) + gaze_point
            gaze_x = gaze_point[0]
            gaze_y = gaze_point[1]
            gazemap[int(round(gaze_y)) - 1][int(round(gaze_x)) - 1] = 1

        # gazemap = ndimage.filters.gaussian_filter(gazemap, 99)
        gazemap = cv2.GaussianBlur(gazemap, ksize=(999, 999), sigmaX=0)
        gazemap -= np.min(gazemap)
        gazemap /= np.max(gazemap)
        gazemap = (gazemap * 255).astype(np.uint8)
        # gazeim = cv2.applyColorMap(gazemap, cv2.COLORMAP_JET)

        save_path = os.path.join(save_dir, str(i).zfill(6) + ".png")
        cv2.imwrite(save_path, gazemap)
            
