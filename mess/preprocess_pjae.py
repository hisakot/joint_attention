import argparse
import csv
import json
import glob
import math
import os
import statistics

import cv2
import numpy as np
from tqdm import tqdm

ROLL = ["NotHuman", "Surgeon", "Assistant", "Anesthesiology", "ScrubNurse", "CirculationNurse", "Visitor"]

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


def generate_gazecone(hs_kpts, H, W, fov_deg=30, cone_length=800, sigma_angle=0.2, sigma_distance=400, max_intensity=1.0):
    gazecone_map = np.ones((H, W), dtype=np.float32)
    for kpt in hs_kpts:
        yv, xv = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        face_kpt = kpt[23:91]
        p1, p2, yaw, pitch, roll = head_direction(face_kpt, H, W)
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

    return gazecone_map

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

def make_face_bbox(hs_kpts, hs_roll, H, W):
    data = dict()
    for i, hs_kpt in enumerate(hs_kpts):
        face_kpt = hs_kpt[23:91]
        min_x, min_y, max_x, max_y = W, H, 0, 0
        for kpt in face_kpt:
            if min_x > kpt[0]:
                min_x = kpt[0]
            if min_y > kpt[1]:
                min_y = kpt[1]
            if max_x < kpt[0]:
                max_x = kpt[0]
            if max_y < kpt[1]:
                max_y = kpt[1]
        radius = max(int(max_x - min_x), int(max_y - min_y))
        p1, p2, yaw, pitch, roll = head_direction(face_kpt, H, W)
        if p1[0] < 0:
            x_center = 0
        elif p1[0] > W:
            x_center = W
        else:
            x_center = p1[0]
        if p1[1] < 0:
            y_center = 0
        elif p1[1] > H:
            y_center = H
        else:
            y_center = p1[1]
        x1 = p1[0]
        y1 = p1[1]
        x2 = p2[0]
        y2 = p2[1]
        x = x2 - x1
        y = y2 - y1
        xy = math.sqrt(x**2+y**2)
        x /= xy
        y /= xy

        roll_num = ROLL.index(hs_roll[i])

        person_data = {"person_idx" : str(i),
                       "head_x_center" : str(x_center),
                       "head_y_center" : str(y_center),
                       "head_radius" : str(radius),
                       "gaze_x" : str(x),
                       "gaze_y" : str(y),
                       "action_num" : str(roll_num),
                       "pred_action_num" : str(roll_num)}
        data.update({str(i) : person_data})
    return data

def make_human_bbox(hs_kpts, H, W, frame_id):
    data = list()
    for i, hs_kpt in enumerate(hs_kpts):
        x_min = W
        y_min = H
        x_max = 0
        y_max = 0
        for kpt in hs_kpt:
            if x_min > kpt[0]:
                x_min = kpt[0]
            if y_min > kpt[1]:
                y_min = kpt[1]
            if x_max < kpt[0]:
                x_max = kpt[0]
            if y_max < kpt[1]:
                y_max = kpt[1]
        w = int(x_max - x_min)
        h = int(y_max - y_min)
        x_min = int(x_min)
        y_min = int(y_min)
        data.append(f"{i} {x_min} {y_min} {w} {h} {frame_id} 0 0 0 passing \n")
    return data

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
    parser.add_argument("--save_video", required=False, action="store_true", help="save file name")
    parser.add_argument("--save_gt", required=False, action="store_true", help="save file name")
    args = parser.parse_args()

    json_paths = glob.glob(os.path.join(args.root, "mmpose/*"))
    # json_paths = glob.glob(os.path.join(args.root, "roll_ann/*"))
    for json_path in tqdm(json_paths):
        if os.path.isfile(json_path):
            video_name = os.path.splitext(os.path.basename(json_path))[0] # ex.) ds_014
            gt_json_path = os.path.join(args.root, "gaze_ann", video_name + ".json")
        else:
            continue

        # video images
        if args.save_video:
            org_frame_paths = glob.glob(os.path.join(args.root, "frames", video_name, "*.png"))
# org_frame_paths = glob.glob(os.path.join(args.root, "frames", "ds005", "*.png"))
            org_frame_paths.sort()
            count = 0
            file_num = 1
            for org_frame_path in org_frame_paths:
                img = cv2.imread(org_frame_path)
                save_dir1 = os.path.join(args.root, "videos")
                if not os.path.exists(save_dir1):
                    os.mkdir(save_dir1)
                save_dir2 = os.path.join(save_dir1, video_name)
                if not os.path.exists(save_dir2):
                    os.mkdir(save_dir2)
                save_dir3 = os.path.join(save_dir2, str(file_num).zfill(6))
                if not os.path.exists(save_dir3):
                    os.mkdir(save_dir3)
                cv2.imwrite(os.path.join(save_dir3, str(count+file_num).zfill(6) + ".png"), img)
                if count == 0:
                    with open(os.path.join(save_dir2, "annotations.txt"), 'a') as f:
                        f.write(str(file_num).zfill(6) + ".png 1\n")
                count += 1
                if count >= 20:
                    file_num += 20
                    count = 0

        # JointAttention GT
        if args.save_gt:
            csv_num = 0
            count = 0
            frame_num = 0
            save_dir = os.path.join(args.root, "annotation_data")
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            with open(gt_json_path) as f:
                data = json.load(f)
                items = data["items"]
                for i, one_frame_ann in enumerate(items):
                    annotations = one_frame_ann["annotations"]
                    if len(annotations) == 0:
                        x1, y1, x2, y2 = 0, 0, 1, 1
                    else:
                        ann = annotations[0] # TODO even if there are some GTs in a frame, using the first GT
                        point = ann["points"]
                        cx = int(round(point[0]))
                        cy = int(round(point[1]))
                        x1 = cx - 1
                        x2 = cx + 1
                        y1 = cy - 1
                        y2 = cy + 1
                        '''
                        bbox = ann["bbox"]
                        x1 = int(bbox[0])
                        y1 = int(bbox[1])
                        x2 = int(x1 + bbox[2])
                        y2 = int(y1 + bbox[3])
                        '''
                    if count % 20 == 0:
                        csv_num += 20
                        count = 0
                    else:
                        frame_num += 1
                    gt_data = [0, x1, y1, x2, y2, count, 0, 0, 0, "Center"]
                    save_path = os.path.join(save_dir, video_name + "_" + str(csv_num-19).zfill(6) + ".csv")
                    with open(save_path, 'a', newline="") as s:
                        writer = csv.writer(s, delimiter=' ')
                        writer.writerow(gt_data)
                    count += 1


        H, W, C = 1920, 3840, 3
        links, instance_info = load_mmpose_json(json_path)
        data_len = len(instance_info)
        for i, instance_info in tqdm(enumerate(instance_info), total=data_len):
            frame_id = instance_info["frame_id"]
            instances = instance_info["instances"]
            hs_kpts = [] # high scored keypoints
            hs_roll = []

            for instance in instances:
                keypoints = instance["keypoints"]
                scores = instance["keypoint_scores"]
                # roll = instance["roll"] # TODO
                roll = "Visitor"
                if sum(score >= 0.5 for score in scores) > 133 / 5:
                    hs_kpts.append(keypoints)
                    hs_roll.append(roll)
            # face bbox
            if i % 20 == 0:
                csv_num = frame_id
            save_dir = os.path.join(args.root, "jae_dataset_bbox_gt")
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_dir = os.path.join(save_dir, video_name)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_dir = os.path.join(save_dir, str(csv_num).zfill(6))
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_path = os.path.join(save_dir, str(frame_id).zfill(6) + ".json")
            frame_data = make_face_bbox(hs_kpts, hs_roll, H, W)
            with open(save_path, 'w') as f:
                json.dump(frame_data, f)

            # human bbox 1
            if i % 20 == 0:
                csv_num = frame_id
            save_dir = os.path.join(args.root, "tracking_annotation")
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_dir = os.path.join(save_dir, video_name)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_dir = os.path.join(save_dir, str(csv_num).zfill(6))
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            frame_data = make_human_bbox(hs_kpts, H, W, frame_id)
            save_path = os.path.join(save_dir, str(frame_id).zfill(6) + ".txt")
            with open(save_path, 'w') as f:
                f.writelines(frame_data)

    # human bbox 2
    frames_dir = glob.glob(os.path.join(args.root, "tracking_annotation", video_name, "*"))
    frames_dir.sort()
    first_txt = 1
    for frame_dir in frames_dir:
        data = list()
        human_num = 0
        max_num = 0
        txt_paths = glob.glob(os.path.join(frame_dir, "*"))
        txt_paths.sort()
        for txt_path in txt_paths:
            with open(txt_path, 'r') as t:
                lines = t.readlines()
                num = len(lines)
                if max_num < num:
                    max_num = num
        while human_num <= max_num:
            for txt_path in txt_paths:
                with open(txt_path, 'r') as t:
                    lines = t.readlines()
                    try:
                        line = lines[human_num]
                        data.append(line)
                    except IndexError:
                        continue
            human_num += 1
        save_path = os.path.join(frame_dir, str(first_txt).zfill(6) + ".txt")
        with open(save_path, 'w') as f:
            f.writelines(data)
        first_txt += 20

        each_txts = glob.glob(os.path.join(frame_dir, "*.txt"))
        for each_txt in each_txts:
            file_name = os.path.basename(each_txt)
            if len(file_name) < 10:
                os.remove(each_txt)
