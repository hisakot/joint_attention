import glob
import math
import os

import cv2
import numpy as np
import torch
# from pytorch_msssim import ssim
from skimage.metrics import structural_similarity as ssim

one_frame_img_paths = glob.glob("data/pred/compare/NoLSTM/*.png")
two_frame_img_paths = glob.glob("data/pred/compare/WithLSTM/*.png")
one_frame_img_paths.sort()
two_frame_img_paths.sort()

H, W = 640, 1280


def calc_ssim(img_paths):
    img_num = len(img_paths)
    ssim_scores = []
    mse_scores = []

    for i in range(img_num - 1):
        b_img = cv2.imread(img_paths[i], 0)
        b_img = cv2.resize(b_img, (W, H))

        a_img = cv2.imread(img_paths[i+1], 0)
        a_img = cv2.resize(a_img, (W, H))

        score = ssim(b_img, a_img, data_range=1)
        ssim_scores.append(score)

        b_center = center_coord(b_img)
        a_center = center_coord(a_img)
        if b_center is None or a_center is None:
            continue
        else:
            mse = math.sqrt((b_center[0] - a_center[0])**2 + (b_center[1] - a_center[1])**2)
            mse_scores.append(mse)

    ssim_total = 0
    for ssim_val in ssim_scores:
        ssim_total += ssim_val
    ssim_total /= len(ssim_scores)

    mse_total = 0
    for mse in mse_scores:
        mse_total += mse
    mse_total /= len(mse_scores)

    return np.mean(ssim_scores), ssim_total, np.mean(mse_scores), mse_total

def center_coord(img):
    thr = int(np.max(img) / 2)
    _, binary = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_brightness = -1
    best_contour = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        mask = np.zeros_like(img, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)
        mean_val = cv2.mean(img, mask=mask)[0]
        if mean_val > max_brightness:
            max_brightness = mean_val
            best_contour = cnt

    M = cv2.moments(best_contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return [cx, cy]
    else:
        return None


one_ssim_np, one_ssim, one_mse_np, one_mse = calc_ssim(one_frame_img_paths)
two_ssim_np, two_ssim, two_mse_np, two_mse = calc_ssim(two_frame_img_paths)

print(one_ssim_np, one_ssim)
print(two_ssim_np, two_ssim)

print(one_mse_np, one_mse)
print(two_mse_np, two_mse)

