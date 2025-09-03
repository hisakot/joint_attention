import glob
import json
import math

import cv2
import numpy as np
from tqdm import tqdm

H = 960 
W = 1920 
F = 1 # frame

img_paths = glob.glob("data/ue/02/MovieRenders/*.png")
img_paths.sort()
# img = cv2.imread("data/ue/MovieRenders/0001.jpeg")
# img = cv2.imread("data/MovieRenders/LevelSequence_OR_0000.png")

camera = {"x" : 600.0, "y" : 400.0, "z" : 200.0}
cam_rot = (0, 180, 0)

def transform(cod_3d, camera):
    cod_3d = np.array([cod_3d["x"], cod_3d["y"], cod_3d["z"]])
    camera = np.array([camera["x"], camera["y"], camera["z"]])
    camera_cod = cod_3d - camera

    pitch, yaw, roll = np.deg2rad(cam_rot)

    R_pitch = np.array([
        [ math.cos(pitch), 0, math.sin(pitch)],
        [ 0,             1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    R_roll = np.array([
        [1, 0,              0             ],
        [0, math.cos(roll),-math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
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

with open("data/ue/02/JSON/Aoi_gaze.json") as f:
    data = json.load(f)
    length = len(data["Structure_gaze"])

    skip = 0
    for i, frame_data in tqdm(enumerate(data["Structure_gaze"]), total=length):
        value = frame_data["gaze"]
        if value["x"] == 0 and value["y"] == 0 and value["z"] == 0:
            skip += 1
            continue
        x, y = transform(value, camera)
        img = cv2.imread(img_paths[i-skip])
        cv2.circle(img, (int(x), int(y)), 10, (0, 0, 255), 2)
        # cv2.putText(img, str(i-skip), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
        cv2.imwrite("data/ue/gaze/" + str(i-skip).zfill(4) + ".png", img)

'''
with open("data/ue/02/JSON/Bernice_face.json") as f:
    data = json.load(f)
    length = len(data["Structure_face"])
    print(len(data["Structure_face"]))

    for i, frame_data in tqdm(enumerate(data["Structure_face"]), total=length):
        img = cv2.imread(img_paths[i])
        cod_3d = data["Structure_face"][i+1]["nose"]
        x, y = transform(cod_3d, camera)

        cv2.circle(img, (int(x), int(y)), 10, (255, 0, 0), 2)
        cv2.putText(img, "f0", (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
        cv2.putText(img, "f0: nose", (100, 100 + i * 70), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0))
        cv2.imwrite("data/ue/gaze/" + str(i).zfill(4) + ".png", img)
'''

'''
with open("data/ue/JSON/Aoi_body.json") as f:
    data = json.load(f)
    print(len(data["Structure_body"]))
    for i, part in enumerate(data["Structure_body"][F]):
        value = data["Structure_body"][F][part]
        x, y = transform(value, camera)
        cv2.circle(img, (int(x), int(y)), 10, (255, 0, 0), 2)
        cv2.putText(img, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
        cv2.putText(img, str(i) + ": " + part, (100, 100 + i * 70), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0))
'''

img_paths = glob.glob("data/ue/gaze/*.png")
img_paths.sort()
with open("data/ue/02/JSON/Bernice_body.json") as f:
    data = json.load(f)
    length = len(data["Structure_body"])
    skip = 0
    for j, data_frame in tqdm(enumerate(data["Structure_body"]), total=length):
        img = cv2.imread(img_paths[j-1])
        for i, part in enumerate(data_frame):
            value = data_frame[part]
            if value["x"] == 0 and value["y"] == 0 and value["z"] == 0:
                skip += 1
                continue
            x, y = transform(value, camera)
            cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), 2)
            # cv2.putText(img, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
        cv2.imwrite("data/ue/gaze/" + str(j-skip).zfill(4) + ".png", img)

'''
with open("data/ue/JSON//Danielle_body.json") as f:
    data = json.load(f)
    print(len(data["Structure_body"]))
    for i, part in enumerate(data["Structure_body"][F]):
        value = data["Structure_body"][F][part]
        x, y = transform(value, camera)
        cv2.circle(img, (int(x), int(y)), 10, (0, 0, 255), 2)
        cv2.putText(img, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
'''

