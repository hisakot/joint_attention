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

    print("yaw",int(yaw),"pitch",int(pitch),"roll",int(roll))

    (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]),
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

with open("data/short_or/mmpose/ds_009_27min-29min.json") as f:
    data = json.load(f)
    img_paths = glob.glob("data/short_or/frames/ds_009_27min-29min/*.png")
    img_paths.sort()
    
    for i, instance_info in enumerate(data["instance_info"]):
        instances = instance_info["instances"]
        org_img = cv2.imread(img_paths[i])

        for instance in instances:
            keypoints = instance["keypoints"]
            face_kpt = keypoints[23:91]
            scores = instance["keypoint_scores"]
            face_scores = scores[23:91]

            new_face_kpt = []
            if sum(score >= 0.5 for score in face_scores) > 68 / 5:
                for face in face_kpt:
                    tpl = tuple([int(xy) for xy in face])
                    cv2.circle(org_img, tpl, 2, (255, 255, 255), thickness=-1)

                p1, p2, yaw, pitch, roll = head_direction(face_kpt, org_img)

                # cv2.arrowedLine(org_img, p1, p2, (0, 0, 255), 5)
                # cv2.putText(org_img, "pitch:" + str(int(pitch)), p1, cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

                pt1_x, pt1_y, pt2_x, pt2_y = face_rectangle(face_kpt, yaw, pitch)
                # cv2.rectangle(org_img, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 255, 255), thickness=2)

        # whole body keypoints
        '''
        H = 1920
        W = 3840
        kptmap = np.zeros((H, W))
        for instance in instances:
            keypoints = instance["keypoints"]
            scores = instance["keypoint_scores"]

            if sum(score >= 0.5 for score in scores) > 133 / 5:
                for kpt in keypoints:
                    gaze_x = kpt[0]
                    gaze_y = kpt[1]
                    kptmap[int(round(gaze_y)) - 1][int(round(gaze_x)) - 1] = 1
        kptmap = kptmap[:, :, np.newaxis]
        kptmap2 = np.concatenate([kptmap, kptmap], 2)
        kptmap = np.concatenate([kptmap2, kptmap], 2)
        kptmap = (kptmap * 255).astype(np.uint8)

        img = cv2.imread(img_paths[i])
        blend_img = cv2.addWeighted(img, 1, kptmap, 1, 0)
        blend_img = cv2.resize(blend_img, (int(W/4), int(H/4)))
        cv2.imwrite("./kpt_blend.png", blend_img)
        cv2.imwrite("./kpt.png", kptmap)
        '''

        img = cv2.resize(org_img, None, fx=0.5, fy=0.5)
        cv2.imwrite("data/face_landmarks/" + str(i).zfill(6) + ".png", img)
        # cv2.imwrite("data/face_landmarks/" + str(44980 - 25 * 19 + i).zfill(6) + ".png", img)
        """
        cv2.imshow("draw", org_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

