import argparse
import csv
import json
import glob
import os
import statistics

import cv2
import numpy as np
from tqdm import tqdm

def img_to_sphere(x_img, y_img, H, W):
    theta = (x_img / W) * 2 * np.pi - np.pi # -π ~ π
    phi = -(y_img / H) * np.pi + (np.pi / 2) # π/2 ~ -π/2

    x_sphere = np.cos(phi) * np.cos(theta)
    y_sphere = np.sin(phi)
    z_sphere = np.cos(phi) * np.sin(theta)
    '''
    theta = (0.5 - y_img / H) * np.pi
    phi = (2 * x_img / W - 1) * np.pi
    x_sphere = np.cos(theta) * np.cos(phi)
    y_sphere = np.cos(theta) * np.sin(phi)
    z_sphere = np.sin(theta)
    '''
    return np.array([x_sphere, y_sphere, z_sphere])

def sphere_to_img(p, H, W):
    x_sphere, y_sphere, z_sphere = p
    theta = np.arctan2(z_sphere, x_sphere) # -π ~ π
    phi = np.arcsin(y_sphere) # -π/2 ~ π/2

    x_img = (theta + np.pi) / (2 * np.pi) * W
    y_img = (-phi + (np.pi / 2)) / np.pi * H
    '''
    x_sphere, y_sphere, z_sphere = p
    theta = np.arcsin(z_sphere)
    phi = np.arctan2(y_sphere, x_sphere)
    x_img = (phi / np.pi + 1) * W / 2
    y_img = (0.5 - theta / np.pi) * H
    '''
    return int(x_img), int(y_img)

def normalize(v):
    norm_v = v / np.linalg.norm(v)
    return norm_v

def get_tangent_vectors(p):
    up_world = np.array([0, 0, 1])
    right = normalize(np.cross(up_world, p))
    up = normalize(np.cross(p, right))
    return right, up

def extend_to_sphere(p, direction):
    dot = np.dot(p, direction)
    return p - 2 * dot * direction

def extend_direction_from_p(p, direction, scale=0.2):
    p = p + scale * direction
    return normalize(p)

def draw_arrow_on_img(img, p, direction, H, W, color=(0, 255, 255)):
    # p_cross = extend_to_sphere(p, direction)
    p_cross = extend_direction_from_p(p, direction, scale=0.3)
    p_img = sphere_to_img(p, H, W)
    p_cross_img = sphere_to_img(p_cross, H, W)
    print(p_img, p_cross_img)
    cv2.arrowedLine(img, p_img, p_cross_img, color, thickness=3, tipLength=0.05)
    return img

def head_direction_3d(face_kpt, H, W):
    # image coordinate(x, y)
    eye_l_img = face_kpt[42]
    eye_r_img = face_kpt[39]
    nose_img = face_kpt[30]
    chin_img = face_kpt[8]

    # image to sphere coordinate
    eye_l_sphere = img_to_sphere(*eye_l_img, H, W)
    eye_r_sphere = img_to_sphere(*eye_r_img, H, W)
    nose_sphere = img_to_sphere(*nose_img, H, W)
    chin_sphere = img_to_sphere(*chin_img, H, W)

    # calculate vector
    lr = normalize(eye_r_sphere - eye_l_sphere)
    down = normalize(chin_sphere - nose_sphere)
    forward = normalize(np.cross(lr, down))

    # gaze target
    dot = np.dot(nose_sphere, forward)
    if dot < 0:
        forward = -forward
    target = nose_sphere - 2 * dot * forward

    return target

def rvec_to_pitch(rvec):
    R, _ = cv2.Rodrigues(rvec)
    pitch = np.arcsin(-R[2, 0])
    return np.degrees(pitch)

def auto_adjust_and_solvePnP(model_points_68, image_points_68, camera_matrix, dist_coeffs,
                             pitch_threshould=-15, nose_shift_per_deg=-1.0,
                             chin_shift_per_deg=-2.0, iterations=3):


    adjusted_model = model_points_68.copy()

    for _ in range(iterations):
        _, rvec, tvec, _ = cv2.solvePnPRansac(
                adjusted_model, image_points_68, camera_matrix, dist_coeffs,
                None, None, False, reprojectionError=4.0, confidence=0.99,
                iterationsCount=500, flags=cv2.SOLVEPNP_ITERATIVE)
        pitch_deg = rvec_to_pitch(rvec)

        if pitch_deg < pitch_threshould:
            pitch_delta = pitch_threshould - pitch_deg
            adjusted_model = model_points_68.copy()
            adjusted_model[30, 2] += nose_shift_per_deg * pitch_delta
            adjusted_model[8, 2] += chin_shift_per_deg * pitch_delta
        else:
            break

    return rvec, tvec

def head_direction_68_solvepnp(face_kpt, H, W):
    image_points_68 = np.array(face_kpt, dtype="double")
    model_points_68 = np.array([
        [0.0, 0.0, 0.0],             # 1
        [0.0, -330.0, -65.0],        # 2
        [0.0, -330.0, -65.0],        # 3
        [0.0, -330.0, -65.0],        # 4
        [0.0, -330.0, -65.0],        # 5
        [0.0, -330.0, -65.0],        # 6
        [0.0, -330.0, -65.0],        # 7
        [0.0, -330.0, -65.0],        # 8 あご先端
        [-225.0, 170.0, -135.0],     # 9
        [-225.0, 170.0, -135.0],     # 10
        [-225.0, 170.0, -135.0],     # 11
        [-225.0, 170.0, -135.0],     # 12
        [-225.0, 170.0, -135.0],     # 13
        [-225.0, 170.0, -135.0],     # 14
        [-225.0, 170.0, -135.0],     # 15
        [225.0, 170.0, -135.0],      # 16
        [225.0, 170.0, -135.0],      # 17
        [225.0, 170.0, -135.0],      # 18
        [225.0, 170.0, -135.0],      # 19
        [225.0, 170.0, -135.0],      # 20
        [225.0, 170.0, -135.0],      # 21
        [225.0, 170.0, -135.0],      # 22
        [-110.0, 220.0, -165.0],     # 23 左眉頭
        [-150.0, 250.0, -165.0],     # 24
        [-185.0, 250.0, -165.0],     # 25
        [-220.0, 220.0, -165.0],     # 26 左眉尻
        [-185.0, 220.0, -165.0],     # 27
        [110.0, 220.0, -165.0],      # 28 右眉頭
        [150.0, 250.0, -165.0],      # 29
        [185.0, 250.0, -165.0],      # 30
        [220.0, 220.0, -165.0],      # 31 右眉尻
        [185.0, 220.0, -165.0],      # 32
        [0.0, 100.0, -100.0],        # 33 鼻根
        [0.0, 0.0, -100.0],          # 34 鼻先
        [0.0, -50.0, -100.0],        # 35
        [0.0, -100.0, -100.0],       # 36
        [-75.0, 100.0, -100.0],      # 37 左目尻
        [-100.0, 150.0, -100.0],     # 38
        [-150.0, 150.0, -100.0],     # 39
        [-175.0, 100.0, -100.0],     # 40
        [-150.0, 75.0, -100.0],      # 41
        [-100.0, 75.0, -100.0],      # 42
        [75.0, 100.0, -100.0],       # 43 右目尻
        [100.0, 150.0, -100.0],      # 44
        [150.0, 150.0, -100.0],      # 45
        [175.0, 100.0, -100.0],      # 46
        [150.0, 75.0, -100.0],       # 47
        [100.0, 75.0, -100.0],       # 48
        [-150.0, -150.0, -125.0],    # 49 左口角
        [-100.0, -175.0, -125.0],    # 50
        [-50.0, -175.0, -125.0],     # 51
        [0.0, -150.0, -125.0],       # 52
        [50.0, -175.0, -125.0],      # 53
        [100.0, -175.0, -125.0],     # 54
        [150.0, -150.0, -125.0],     # 55 右口角
        [100.0, -125.0, -125.0],     # 56
        [50.0, -125.0, -125.0],      # 57
        [0.0, -125.0, -125.0],       # 58
        [-50.0, -125.0, -125.0],     # 59
        [-100.0, -125.0, -125.0],    # 60
        [-100.0, -200.0, -125.0],    # 61
        [-50.0, -225.0, -125.0],     # 62
        [0.0, -225.0, -125.0],       # 63
        [50.0, -225.0, -125.0],      # 64
        [100.0, -200.0, -125.0],     # 65
        [50.0, -175.0, -125.0],      # 66
        [0.0, -175.0, -125.0],       # 67
        [-50.0, -175.0, -125.0]      # 68
    ], dtype='double')

    fov_x_deg = 90
    fov_y_deg = 60
    fov_x_rad = np.deg2rad(fov_x_deg)
    fov_y_rad = np.deg2rad(fov_y_deg)
    fx = W / (2 * np.tan(fov_x_rad / 2))
    fy = H / (2 * np.tan(fov_y_rad / 2))
    center = (W / 2, H / 2)
    camera_matrix = np.array([[fx, 0, center[0]],
                              [0, fy, center[1]],
                              [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    rvec, tvec = auto_adjust_and_solvePnP(model_points_68, image_points_68,
                                          camera_matrix, dist_coeffs,
                                          pitch_threshould=5,
                                          nose_shift_per_deg=-4.0,
                                          chin_shift_per_deg=-8.0,
                                          iterations=3)
    R, _ = cv2.Rodrigues(rvec)
    face_direction = R @ np.array([0, 0, 1])
    face_direction /= np.linalg.norm(face_direction)

    return face_direction

def vector_to_equirectangular(v):
    x, y, z = v
    lon = np.arctan2(x, z)
    lat = np.arcsin(-y)
    return np.degrees(lon), np.degrees(lat)

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

    if pitch > 0:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if dy < 0:
            p2 = (int(p1[0] - dx), int(p1[1] - dy))

    return p1, p2, yaw, pitch, roll

def intersect_sphere(v1, dir_vec):
    a = np.dot(dir_vec, dir_vec)
    b = 2 * np.dot(v1, dir_vec)
    c = np.dot(v1, v1) - 1.0 # radius=1
    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        return None
    t = (-b + np.sqrt(discriminant)) / (2 * a)
    intersection = v1 + t * dir_vec
    return intersection

def get_intersection_2d(p1, p2, H, W):
    v1 = img_to_sphere(*p1, H, W)
    up = np.array([0, 1, 0])
    if np.allclose(v1, up):
        up = np.array([1, 0, 0])
    tangent_x = np.cross(up, v1)
    tangent_x = tangent_x / np.linalg.norm(tangent_x)
    tangent_y = np.cross(v1, tangent_x)
    tangent_y = tangent_y / np.linalg.norm(tangent_y)

    delta2d = np.array([p2[0] - p1[0], p2[1] - p1[1]])

    dir_3d = delta2d[0] * tangent_x + delta2d[1] * tangent_y
    if np.dot(v1, dir_3d) < 0:
        dir_3d = -dir_3d
    dir_3d = dir_3d / np.linalg.norm(dir_3d)

    '''
    P(t) = p1 + t * dir_3d
    |P(t)|**2 = 1
    |p1 + t * dir_3d} ** 2 = 1
    a * t**2 + b * t + c = 0
    a = 1
    b = 2 * (p1 * dir_3d)
    c = |p1|**2 - 1 = 0
    t**2 + 2 * (p1 * dir_3d) * t = 0
    '''
    a = 1.0
    b = 2.0 * np.dot(v1, dir_3d)
    c = 0.0 # on the sphere

    discriminant = b ** 2 - 4 * a * c
    sqrt_disc = np.sqrt(discriminant)

    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    if t2 < 1e-6:
        print("t2 is smaller")
        intersection_3d = v1 + 0.1 * dir_3d
        intersection_3d /= np.linalg.norm(intersection_3d)
        intersection_2d = sphere_to_img(intersection_3d, H, W)
    else:
        intersection_3d = v1 + t2 * dir_3d
        intersection_3d /= np.linalg.norm(intersection_3d)
        intersection_2d = sphere_to_img(intersection_3d, H, W)
    print("----------------------------------------")

    return intersection_2d

def get_intersection_2d_2(p1, p2, H, W):
    v1 = img_to_sphere(*p1, H, W)
    v2 = img_to_sphere(*p2, H, W)
    dir_3d = v2 - v1
    dir_3d = dir_3d / np.linalg.norm(dir_3d)

    '''
    P(t) = p1 + t * dir_3d
    |P(t)|**2 = 1
    |p1 + t * dir_3d} ** 2 = 1
    a * t**2 + b * t + c = 0
    a = 1
    b = 2 * (p1 * dir_3d)
    c = |p1|**2 - 1 = 0
    t**2 + 2 * (p1 * dir_3d) * t = 0
    '''
    a = 1.0
    b = 2.0 * np.dot(v1, dir_3d)
    c = 0.0 # on the sphere

    discriminant = b ** 2 - 4 * a * c
    sqrt_disc = np.sqrt(discriminant)

    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    if t2 < 1e-6:
        print("t2 is smaller")
        intersection_3d = v1 + 0.1 * dir_3d
        intersection_3d /= np.linalg.norm(intersection_3d)
        intersection_2d = sphere_to_img(intersection_3d, H, W)
    else:
        intersection_3d = v1 + t2 * dir_3d
        intersection_3d /= np.linalg.norm(intersection_3d)
        intersection_2d = sphere_to_img(intersection_3d, H, W)
    print("----------------------------------------")

    return intersection_2d


def generate_gaze_heatmap(heatmap, p1, p2, sigma_angle=0.2, sigma_distance=500):
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
    distance_weight = np.exp(- (distances ** 2) / (2 * sigma_distance ** 2))
    # distance_weight = 1

    heatmap += angle_weight * distance_weight

    return heatmap

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some integers")
    parser.add_argument("--mmpose_dir", required=True, type=str)
    parser.add_argument("--save_dir", required=True, type=str)
    args = parser.parse_args()

    json_paths = glob.glob(os.path.join(args.mmpose_dir, "*"))
    for json_path in tqdm(json_paths):
        if os.path.isfile(json_path):
            file_name = os.path.splitext(os.path.basename(json_path))[0] # ex.) ds014
        else:
            continue

        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        save_gazecone_dir = os.path.join(args.save_dir, file_name)
        if not os.path.exists(save_gazecone_dir):
            os.mkdir(save_gazecone_dir)
        else:
            print(save_gazecone_dir, " is exists. REWRITE!")

        with open(json_path) as f:
            data = json.load(f)
            
            data_len = len(data["instance_info"])
            for i, instance_info in tqdm(enumerate(data["instance_info"]), total=data_len):
                instances = instance_info["instances"]
                H, W, C = 1920, 3840, 3
                heatmap = np.zeros((H, W), dtype=np.float32)
                img = cv2.imread("data/test/frames/ds_005/" + str(i*100).zfill(6) + ".png")

                for j, instance in enumerate(instances):
                    keypoints = instance["keypoints"]
                    face_kpt = keypoints[23:91]
                    scores = instance["keypoint_scores"]
                    face_scores = scores[23:91]

                    if sum(score >= 0.5 for score in face_scores) > 68 / 5:
                        for idx, (x, y) in enumerate(face_kpt):
                            cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)

                        p1, p2, _, _, _ = head_direction(face_kpt, H, W)
                        cv2.arrowedLine(img, p1, p2, (255, 0, 0), thickness=3, tipLength=0.5)
                        print(p1, p2, np.sqrt((p2[0]-p1[0])**2 +(p2[1]-p1[1])**2))
                        '''
                        new_p2 = get_intersection_2d_2(p1, p2, H, W)
                        cv2.arrowedLine(img, p1, new_p2, (0, 0, 255), thickness=3, tipLength=0.5)

                        target = head_direction_3d(face_kpt, H, W)
                        nose_img = tuple([int(p) for p in face_kpt[30]])
                        target_img = sphere_to_img(target, H, W)
                        print(nose_img, target_img)
                        cv2.arrowedLine(img, nose_img, target_img, (0, 0, 255), thickness=3, tipLength=0.05)

                        nose_sphere = img_to_sphere(*face_kpt[30], H, W)
                        right, up = get_tangent_vectors(nose_sphere)
                        move_vec = right * 1.0 + up * 0.0
                        move_vec = normalize(move_vec)
                        img = draw_arrow_on_img(img, nose_sphere, move_vec, H, W)

                        face_direction = head_direction_68_solvepnp(face_kpt, H, W)
                        target_img = sphere_to_img(face_direction, H, W)
                        cv2.arrowedLine(img, p1, target_img, (0, 0, 255), thickness=3, tipLength=0.05)
                        '''

                        '''
                        p1_sphere = img_to_sphere(*p1, H, W)
                        p2_sphere = img_to_sphere(*p2, H, W)
                        dir_vec = p2_sphere - p1_sphere
                        dir_vec = dir_vec / np.linalg.norm(dir_vec) # normalize

                        intersection = intersect_sphere(p1_sphere, dir_vec)
                        if intersection is not None:
                            intersection_img = sphere_to_img(intersection, H, W)
                            cv2.arrowedLine(img, p1, intersection_img, (0, 0, 255),
                                            thickness=3, tipLength=0.05)
                        '''


                '''
                img = cv2.resize(img, None, fx=0.5, fy=0.5)
                cv2.imwrite("data/test/gazearea/ds_005/" + str(i) + ".png", img)
                cv2.imshow("img", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                        heatmap = generate_gaze_heatmap(heatmap, p1, intersection_img)
                    # normalize heatmap [0, 1]
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
                    heatmap = (heatmap * 255).astype(np.float32)
                    cv2.imwrite(os.path.join(save_gazecone_dir, str(i).zfill(6)) + ".png", heatmap)
                '''

