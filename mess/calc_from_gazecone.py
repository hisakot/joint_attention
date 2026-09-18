import glob
import math
import os

import cv2
import numpy as np
from tqdm import tqdm

cone_paths = glob.glob("data/short_test/gazearea/009_5min_10/*.png")
cone_paths.sort()

for i, cone_path in tqdm(enumerate(cone_paths), total=len(cone_paths)):
    cone = cv2.imread(cone_path, cv2.IMREAD_GRAYSCALE)
    max_value = cone.max()
    threshold = max_value * 0.5

    '''
    _, binary = cv2.threshold(cone, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, h = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = np.zeros_like(cone)
    cv2.drawContours(output, contours, -1, 255, thickness=cv2.FILLED)
    '''

    output = np.where(cone >= threshold, 255, 0).astype(np.uint8)

    cv2.imwrite("data/pred/geometric_from_cone/"+str(i+1).zfill(6)+".png", output)
