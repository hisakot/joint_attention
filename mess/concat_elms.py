import cv2
import numpy as np

import glob
import os

img_paths = glob.glob("data/test/frames/*/*.png")
img_paths.sort()
kpt_paths = glob.glob("data/test/kptmap/*/*.png")
kpt_paths.sort()
gazecone_paths = glob.glob("data/test/gazecone_que/*/*.png")
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
    kpt_map = np.concatenate([zeros, kpt_map, zeros], axis=2)

    gazecone_map = cv2.imread(gazecone_paths[i], 0)
    gazecone_map = cv2.resize(gazecone_map, (W, H))
    gazecone_map = gazecone_map[:, :, np.newaxis]
    gazecone_map = np.concatenate([zeros, gazecone_map, gazecone_map], axis=2)

    # result = cv2.addWeighted(img, 0.8, kpt_map, 1, 0)
    result = cv2.addWeighted(img, 1, gazecone_map, 1, 0)

    cv2.imwrite("data/test/pred/kpt_gaze_que/" + str(i).zfill(6) + ".png", result)

