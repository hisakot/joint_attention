import argparse
import csv
import json
import glob
import os
import statistics

import cv2
import numpy as np
from tqdm import tqdm

def head_direction(face_kpt, H, W):
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

    focal_length = W
    center = (W // 2, H // 2) # center of face

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

    '''
    print("yaw",int(yaw),"pitch",int(pitch),"roll",int(roll))
    '''

    (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 2000.0)]),
                                              rotation_vector, translation_vector,
                                              camera_matrix, dist_coeffs)
    
    """
    for p in image_points:
        cv2.drawMarker(org_img, (int(p[0]), int(p[1])), (0.0, 1.409845, 255), 
                       markerType=cv2.MARKER_CROSS, thickness=1)
    """

    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    return p1, p2, yaw, pitch, roll

def generate_gaze_heatmap(heatmap, p1, p2, intensity,  sigma_angle=0.2, sigma_distance=1000):
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

    distances = np.sqrt((x_indices - p1[0])**2 + (y_indices - p1[1])**2)

    # gaze strength based on gauss
    angle_weight = np.exp(- (theta_diff ** 2) / (2 * sigma_angle ** 2))
    distance_weight = np.exp(- (distances ** 2) / (2 * sigma_distance ** 2)) # if closeer is white
    distance_weight = 1

    heatmap += intensity * angle_weight * distance_weight

    return heatmap

def calc_intensity(ja_rate):
    if ja_rate == "0.0:own_work":
        intensity = 0.2
    elif ja_rate == "0.5:ja_sub":
        intensity = 0.5
    elif ja_rate == "1.0:ja_core":
        intensity = 1.0
    return intensity

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some integers")
    parser.add_argument("--mmpose_dir", required=True, type=str)
    parser.add_argument("--roll_dir", required=True, type=str)
    parser.add_argument("--save_dir", required=True, type=str)
    args = parser.parse_args()

    mmpose_paths = glob.glob(os.path.join(args.mmpose_dir, "*"))
    mmpose_paths.sort()
    roll_paths = glob.glob(os.path.join(args.roll_dir, "*"))
    roll_paths.sort()
    for mmpose_path in tqdm(mmpose_paths):
        if os.path.isfile(mmpose_path):
            file_name = os.path.splitext(os.path.basename(mmpose_path))[0] # ex.) results_ds_014
        else:
            continue
        roll_path = mmpose_path.replace("mmpose/results_ds_", "roll_ann/ds")
        if not os.path.exists(roll_path):
            continue

        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        save_gazecone_dir = os.path.join(args.save_dir, file_name)
        if not os.path.exists(save_gazecone_dir):
            os.mkdir(save_gazecone_dir)
        else:
            print(save_gazecone_dir, " is exists")
            continue

        with open(mmpose_path) as f:
            data = json.load(f)
            data_len = len(data["instance_info"])

            with open(roll_path) as r:
                ann_data = json.load(r)
                ann_info = ann_data["ann_info"]
                ann_idx = 0
            
                for i, instance_info in tqdm(enumerate(data["instance_info"]), total=data_len):
                    instances = instance_info["instances"]
                    H, W, C = 1920, 3840, 3
                    heatmap = np.zeros((H, W), dtype=np.float32)

                    for instance in instances:
                        ann = ann_info[ann_idx]
                        frame_id = ann["frame_id"]
                        intensity = calc_intensity(ann["ja_rate"])
                        if frame_id - 1 == i:
                            keypoints = instance["keypoints"]
                            face_kpt = keypoints[23:91]
                            scores = instance["keypoint_scores"]
                            face_scores = scores[23:91]

                            if sum(score >= 0.5 for score in face_scores) > 68 / 5:
                                p1, p2, yaw, pitch, roll = head_direction(face_kpt, H, W)
                                heatmap = generate_gaze_heatmap(heatmap, p1, p2, intensity)
                            ann_idx += 1
                    # normalize heatmap [0, 1]
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
                    heatmap = (heatmap * 255).astype(np.float32)
                    cv2.imwrite(os.path.join(save_gazecone_dir, str(i).zfill(6)) + ".png", heatmap)

