import argparse
import glob
import json
import os

import cv2
import numpy as np
from scipy import ndimage
from tqdm import tqdm

def load_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
        targets = data["items"]

        return targets

            
def gaussian_blur(gazemap, ksize):
    # gazemap = ndimage.filters.gaussian_filter(gazemap, 99)
    gazemap = cv2.GaussianBlur(gazemap, ksize=(ksize, ksize), sigmaX=0)
    gazemap -= np.min(gazemap)
    gazemap /= np.max(gazemap)
    gazemap = (gazemap * 255).astype(np.uint8)
    # gazeim = cv2.applyColorMap(gazemap, cv2.COLORMAP_JET)
    return gazemap

def transform(cod_3d, camera, cam_rot, H, W):
    cod_3d = np.array([cod_3d["x"], cod_3d["y"], cod_3d["z"]])
    camera = np.array([camera["x"], camera["y"], camera["z"]])
    camera_cod = cod_3d - camera

    pitch, yaw, roll = np.deg2rad(cam_rot)

    R_pitch = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [ 0,             1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    R_roll = np.array([
        [1, 0,              0             ],
        [0, np.cos(roll),-np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])

    # World to camera coordinate
    R = R_roll @ R_pitch @ R_yaw
    local = R.T @ camera_cod
    
    d = local / np.linalg.norm(local, ord=2)

    phi = np.arctan2(d[1], d[0])
    theta = np.arcsin(d[2])

    x = (phi + np.pi) / (2 * np.pi) * W
    y = (1 - (theta + np.pi/2) / np.pi) * H

    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", required=True, type=str)
    args = parser.parse_args()

    if args.data_type == "ue":
        H = 960
        W = 1920
        save_dir = "data/ue/gt_heatmap_1ch/ds_ue01"
        camera = {"x" : 500.0, "y" : 450.0, "z" : 160.0}
        cam_rot = (0, 180, 0)

        with open("data/ue/JSON/Aoi_gaze_ue01.json") as f:
            data = json.load(f)
            length = len(data["Structure_gaze"])

            frame_num = 0
            for i, frame_data in tqdm(enumerate(data["Structure_gaze"]), total=length):
                value = frame_data["gaze"]
                if value["x"] == 0 and value["y"] == 0 and value["z"] == 0:
                    continue
                if frame_num % 10 == 0:
                    x, y = transform(value, camera, cam_rot, H, W)
                    gazemap = np.zeros((H, W))
                    gazemap[int(round(y)) - 1][int(round(x)) - 1] = 1
                    gazemap = gaussian_blur(gazemap, ksize=299)

                    save_path = os.path.join(save_dir, str(frame_num).zfill(6) + ".png")
                    cv2.imwrite(save_path, gazemap)
                frame_num += 1

    elif args.data_type == "real_bbox":
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
                gazemap = gaussian_blur(gazemap, ksize=999)

                save_path = os.path.join(save_dir, str(i).zfill(6) + ".png")
                cv2.imwrite(save_path, gazemap)
    elif args.data_type == "real_point":
        W = 1920
        H = 960
        gaze_ann_paths = glob.glob("data/short_or/annotations/*.json")
        for gaze_ann_path in gaze_ann_paths:
            data = load_json(gaze_ann_path)

            video_name = os.path.splitext(os.path.basename(gaze_ann_path))[0]
            save_dir = "data/short_or/gt_heatmap_1ch/" + video_name
            if os.path.exists(save_dir):
                print(save_dir + " is exits")
                continue
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            
            for i, one_frame_ann in enumerate(data):
                ann_in_img = one_frame_ann["annotations"]
                gazemap = np.zeros((H, W))
                for ann in ann_in_img:
                    point = ann["points"] # (f.f, f.f)
                    gazemap[int(round(point[1]))][int(round(point[0]))] = 1
                gazemap = gaussian_blur(gazemap, ksize=499)
                save_path = os.path.join(save_dir, str(i).zfill(6) + ".png")
                cv2.imwrite(save_path, gazemap)

if __name__ == '__main__':
    main()
