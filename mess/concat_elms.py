import cv2
import numpy as np

import glob
import os

img_paths = glob.glob("data/Fig/frame/*.png")
img_paths.sort()
kpt_paths = glob.glob("data/Fig/kptmap/ds_ue_01/*.png")
kpt_paths.sort()
gazecone_paths = glob.glob("data/ue/train/gazearea/ds_ue_01/*.png")
gazecone_paths.sort()

H = 480
W = 960

for i, img_path in enumerate(img_paths):
    zeros = np.zeros((H, W), dtype=np.uint8)
    zeros = zeros[:, :, np.newaxis]

    img = cv2.imread(img_path)
    img = cv2.resize(img, (W, H))

    kpt_map = cv2.imread(kpt_paths[i], 0)
    kpt_map = cv2.resize(kpt_map, (W, H))
    kpt_map = kpt_map[:, :, np.newaxis]
    kpt_map = np.concatenate([kpt_map, zeros, kpt_map], axis=2) # mazenta

    gazecone_map = cv2.imread(gazecone_paths[i], 0)
    gazecone_map = cv2.resize(gazecone_map, (W, H))
    # if color
    gazecone_map = cv2.applyColorMap(gazecone_map, cv2.COLORMAP_JET)
    cv2.imwrite("data/ue/videos/gazecolor/" + str(i+1).zfill(6) + ".png", gazecone_map)
    # if no color
    # gazecone_map = gazecone_map[:, :, np.newaxis]
    # gazecone_map = np.concatenate([zeros, gazecone_map, gazecone_map], axis=2)

    result = cv2.addWeighted(img, 0.9, kpt_map, 1, 0)
    result = cv2.addWeighted(result, 1, gazecone_map, 1, 0)

    kpt_gaze = cv2.addWeighted(kpt_map, 1, gazecone_map, 1, 0)

    cv2.imwrite("data/Fig/gazemult/" + str(i+1).zfill(6) + ".png", result)
    cv2.imwrite("data/Fig/kpt_gaze/" + str(i+1).zfill(6) + ".png", kpt_gaze)

