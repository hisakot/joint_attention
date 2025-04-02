import csv
import json
import glob
import statistics

import cv2
import numpy as np

def face_rectangle(face_kpt, yaw, pitch):
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

def head_direction(face_kpt, org_img):
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

    size = org_img.shape
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

def generate_gaze_heatmap(heatmap, p1, p2, sigma_angle=0.2, sigma_distance=1000):
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
    distance_weight = 1

    heatmap += angle_weight * distance_weight

    return heatmap

with open("data/train/mmpose/results_ds_014.json") as f:
    data = json.load(f)
    img_paths = glob.glob("data/train/frames/ds014/*.png")
    img_paths.sort()
    
    for i, instance_info in enumerate(data["instance_info"]):
        instances = instance_info["instances"]
        org_img = cv2.imread(img_paths[i])
        H, W, C = org_img.shape
        gazeline_map = np.zeros((H, W, C))
        heatmap = np.zeros((H, W), dtype=np.float32)

        for instance in instances:
            keypoints = instance["keypoints"]
            face_kpt = keypoints[23:91]
            scores = instance["keypoint_scores"]
            face_scores = scores[23:91]

            if sum(score >= 0.5 for score in face_scores) > 68 / 5:
                for face in face_kpt:
                    tpl = tuple([int(xy) for xy in face])
                    cv2.circle(org_img, tpl, 2, (0, 255, 0), thickness=-1)

                p1, p2, yaw, pitch, roll = head_direction(face_kpt, org_img)

                cv2.line(gazeline_map, p1, p2, (225, 225, 255), thickness=10)
                heatmap = generate_gaze_heatmap(heatmap, p1, p2)
        # normalize heatmap [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
        '''
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        resize = cv2.resize(gazeline_map, None, fx=0.2, fy=0.2)
        cv2.imshow("gazeline_map", resize)
        '''
        heatmap = (heatmap * 255).astype(np.uint8)
        resize = cv2.resize(heatmap, None, fx=0.2, fy=0.2)
        cv2.imshow("heatmapmap", resize)
        cv2.imwrite("gazecone.png", heatmap)
        cv2.waitKey(0)
        org_img = cv2.resize(org_img, None, fx=0.2, fy=0.2)
        cv2.imshow("org_img", org_img)
        cv2.imwrite("org_img.png", org_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit()
        cv2.imwrite("data/train/gazeline_map/"+os.path.basename(img_paths[i]), gazeline_map)

