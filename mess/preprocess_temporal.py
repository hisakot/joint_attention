import argparse
import csv
import json
import glob
import math
import os
from collections import deque
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

def get_body_forward(kpts):
    kpt_num = [5, 6, 11, 12]
    def get_point(num):
        return np.array(kpts[num]) if kpts[num] is not None else None

    def normalize(v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    ls = get_point(kpt_num[0]) # left shouldrt
    rs = get_point(kpt_num[1]) # right shoulder
    lh = get_point(kpt_num[2]) # left hip
    rh = get_point(kpt_num[3]) # right hip

    shoulder_center = None
    if ls is not None and rs is not None:
        shoulder_center = (ls + rs) / 2
    elif ls is not None:
        shoulder_center = ls
    elif rs is not None:
        shoulder_center = rs
    else:
        return None

    hip_center = None
    if lh is not None and rh is not None:
        hip_center = (lh + rh) / 2
    elif lh is not None:
        hip_center = lh
    elif rh is not None:
        hip_center = rh
    else:
        return None

    if shoulder_center is None or hip_center is None:
        return None

    if ls is not None and rs is not None:
        shoulder_vec = np.array(rs) - np.array(ls)
        shoulder_vec = normalize(shoulder_vec)

        body_forward = np.array([-shoulder_vec[1], shoulder_vec[0]])
        body_forward = normalize(body_forward)

        spine_vec = normalize(hip_center - shoulder_center)
        if np.dot(body_forward, spine_vec) < 0:
            body_forward = -body_forward
        return body_forward
    else:
        spine_vec = normalize(hip_center - shoulder_center)
        return spine_vec


def generate_gazecone(hs_kpts, H, W, gazeque, fov_deg=30, cone_length=800, sigma_angle=0.2, sigma_distance=400, max_intensity=1.0):
    gazecone_map = np.ones((H, W), dtype=np.float32)
    after_pt = []
    for i, kpt in enumerate(hs_kpts):
        yv, xv = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        face_kpt = kpt[23:91]
        p1, p2, yaw, pitch, roll = head_direction(face_kpt, H, W)

        # TODO!!!!! how to compare before pts
        for before_pt in gazeque:
            min_distance = float('inf')
            for b_pt in before_pt:
                b_p1 = b_pt[0]
                b_p2 = b_pt[1]
                distance = (p1[0] - b_p1[0])**2 + (p1[1] - b_p1[1])**2
                if min_distance > distance:
                    min_distance = distance
                    close = b_pt
            a_vec = np.array([p2[0]-p1[0], p2[1]-p1[1]])
            b_vec = np.array([close[1][0]-close[0][0], close[1][1]-close[0][1]])
            dot = a_vec@b_vec
            if dot < 0:
                x = 2 * p1[0] - p2[0]
                y = 2 * p1[1] - p2[1]
                p2 = (x, y)
        after_pt.append([p1, p2])
        '''
        body_forward = get_body_forward(kpt)
        face_vec = [p2[0] - p1[0], p2[1] - p1[1]]
        face_forward = (face_vec) / np.linalg.norm(face_vec)
        if body_forward is not None:
            similality = np.dot(face_forward, body_forward)
            if similality < 0:
                p2 = (2 * p1[0] - p2[0], 2 * p1[1] - p2[1])
        '''

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        norm = np.sqrt(dx**2 + dy**2)
        dx /= norm
        dy /= norm

        direction_angle = math.atan2(dy, dx)
        fov_rad = np.deg2rad(fov_deg / 2)

        dx_wrap = ((xv - p1[0] + W // 2) % W) - W // 2 # wrap-around
        dy_grid = yv - p1[1]
        distance = np.sqrt(dx_wrap**2 + dy_grid**2)

        # difference of angle
        angle = np.arctan2(dy_grid, dx_wrap)
        angle_diff = np.arctan2(np.sin(angle - direction_angle), np.cos(angle - direction_angle))

        # masked gaze range
        # in_cone = (np.abs(angle_diff) <= fov_rad) & (distance <= cone_length)

        # reducing by angle and distance
        angle_weight = np.exp(- (angle_diff ** 2) / (2 * sigma_angle ** 2))
        distance_weight = np.exp(- (distance ** 2) / (2 * sigma_distance ** 2))
        weight = max_intensity * angle_weight * distance_weight
        weight += 1
        # weight[~in_cone] = 0

        gazecone_map *= weight

    # normalize heatmap [0, 1]
    gazecone_map = (gazecone_map - gazecone_map.min()) / (gazecone_map.max() - gazecone_map.min() + 1e-6) # TODO 
    gazecone_map = (gazecone_map * 255).astype(np.uint8)
    gazecone_map = gazecone_map[:, :, np.newaxis]

    gazeque.appnd(after_pt)

    return gazecone_map, after_pt

def generate_pose(hs_kpts, H, W, links, sigma=3):
    pose_map = np.zeros((H, W, 1), dtype=np.uint8)
    for kpt133 in hs_kpts:
        for p in kpt133:
            x = int(p[0])
            y = int(p[1])
            if x >= 0 and x < W and y >= 0 and y < H:
                cv2.circle(pose_map, (x, y), radius=6, color=255, thickness=-1)
        for link in links:
            p1 = kpt133[link[0]]
            p2 = kpt133[link[1]]
            p1x = int(p1[0])
            p1y = int(p1[1])
            p2x = int(p2[0])
            p2y = int(p2[1])
            cv2.line(pose_map, (p1x, p1y), (p2x, p2y), 255, thickness=3)
    pose_map = np.max(pose_map, axis=2)
    pose_map = pose_map[:, :, np.newaxis]

    return pose_map

def load_mmpose_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
        meta_info = data["meta_info"]
        links = meta_info["skeleton_links"]

        instance_info = data["instance_info"]

        return links, instance_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some integers")
    parser.add_argument("--root", required=True, type=str) # data/train/
    parser.add_argument("--save_gazecone", required=False, type=str, help="save file name")
    parser.add_argument("--save_posemap", required=False, type=str, help="save file name")
    args = parser.parse_args()

    json_paths = glob.glob(os.path.join(args.root, "mmpose/*"))
    for json_path in tqdm(json_paths):
        if os.path.isfile(json_path):
            file_name = os.path.splitext(os.path.basename(json_path))[0] # ex.) ds014
        else:
            continue

        H, W, C = 1920, 3840, 3
        links, instance_info = load_mmpose_json(json_path)
        data_len = len(instance_info)
        gazeque = deque(maxlen=5)
        for i, instance_info in tqdm(enumerate(instance_info), total=data_len):
            frame_id = instance_info["frame_id"]
            instances = instance_info["instances"]
            hs_kpts = [] # high scored keypoints

            for instance in instances:
                keypoints = instance["keypoints"]
                scores = instance["keypoint_scores"]
                if sum(score >= 0.5 for score in scores) > 133 / 5:
                    hs_kpts.append(keypoints)
            if args.save_gazecone:
                save_dir = os.path.join(args.root, args.save_gazecone)
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                save_gazecone_dir = os.path.join(save_dir, file_name)
                if not os.path.exists(save_gazecone_dir):
                    os.mkdir(save_gazecone_dir)
                gazecone_map, after_pt = generate_gazecone(hs_kpts, H, W, 
                                                          gazeque,
                                                          sigma_angle=0.2,
                                                          sigma_distance=500)
                '''
                for pt in after_pt:
                    cv2.arrowedLine(gazecone_map, pt[0], pt[1], color=256, thickness=10)
                if len(before_pt) != 0:
                    for pt in before_pt:
                        cv2.arrowedLine(gazecone_map, pt[0], pt[1], color=256, thickness=3)
                '''
                cv2.imwrite(os.path.join(save_gazecone_dir, str(frame_id).zfill(6)) + ".png", gazecone_map)
            if args.save_posemap:
                save_dir = os.path.join(args.root, args.save_posemap)
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                save_posemap_dir = os.path.join(save_dir, file_name)
                if not os.path.exists(save_posemap_dir):
                    os.mkdir(save_posemap_dir)
                pose_map = generate_pose(hs_kpts, H, W, links, sigma=3)
                cv2.imwrite(os.path.join(save_posemap_dir, str(frame_id).zfill(6)) + ".png", pose_map)

