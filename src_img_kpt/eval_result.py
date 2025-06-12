import glob
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

gt_paths = glob.glob("data/test/gt_heatmap_1ch/*/*.png")
gt_paths.sort()
pred_paths = glob.glob("data/test/pred/result1/*.png")
pred_paths.sort()
size = len(gt_paths)
H = 640
W = 1280

x, y, xy, auc_sum = 0, 0, 0, 0
list_x = []
list_y = []
list_xy = []
thr30 = 0
thr60 = 0
thr90 = 0
for i, gt_path in tqdm(enumerate(gt_paths), total=len(gt_paths)):
    # GT
    gt = cv2.imread(gt_path, 0)
    gt = cv2.resize(gt, (W, H))
    gt = gt.astype(np.float64)
    gt /= 255.
    gt_max = np.max(gt)
    gt_argmax = np.unravel_index(np.argmax(gt), gt.shape)
    gt_flat = gt.reshape(-1)
    gt_flat = np.where(gt_flat > 0, 1, 0)

    # PRED
    pred_path = pred_paths[i]
    pred = cv2.imread(pred_path, 0)
    pred = cv2.resize(pred, (W, H))
    pred = pred.astype(np.float64)
    pred /= 255.
    pred_max = np.max(pred)
    pred_argmax = np.unravel_index(np.argmax(pred), pred.shape)
    pred_flat = pred.reshape(-1)

    # if using pred moment
    pred = cv2.imread(pred_path, 0)
    pred = cv2.resize(pred, (W, H))
    pred = pred[:, :, np.newaxis]
    _, pred_binary = cv2.threshold(pred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pred_contours, _ = cv2.findContours(pred_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dist_min = [0, 0, math.sqrt(W**2 + H**2)]
    for cnt in pred_contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            dist_x = abs(cx - gt_argmax[1])
            if dist_x > W / 2:
                dist_x = W - dist_x
            dist_y = abs(cy - gt_argmax[0])
            dist_xy = math.sqrt(dist_x^2 + dist_y^2)
            if dist_min[2] > dist_xy:
                dist[0] = dist_x
                dist[1] = dist_y
                dist_min[2] = dist_xy
    x += dist_min[0]
    y += dist_min[1]
    xy += dist_min[2]
    list_x.append(dist_min[0])
    list_y.append(dist_min[1])
    list_xy.append(dist_min[2])
    if dist_min[2] <= 30:
        thr30 += 1
    if dist_min[2] <= 60:
        thr60 += 1
    if dist_min[2] <= 90:
        thr90 += 1

    # if using pred argmax
    '''
    result = cv2.imread(pred_path, 1)
    result = cv2.resize(result, (W, H))
    result = cv2.circle(result, (pred_argmax[1], pred_argmax[0]), 10, (255, 0, 0), 3)
    result = cv2.circle(result, (gt_argmax[1], gt_argmax[0]), 10, (0, 0, 255), 3)
    cv2.imwrite("data/test/pred/result/" + str(i).zfill(6) + ".png", result)
    dist_x = abs(pred_argmax[1] - gt_argmax[1])
    if dist_x > W / 2:
       dist_x = W - dist_x
    dist_y = abs(pred_argmax[0] - gt_argmax[0])
    dist_xy = math.sqrt(dist_x^2 + dist_y^2)
    x += dist_x
    y += dist_y
    xy += dist_xy
    list_x.append(dist_x)
    list_y.append(dist_y)
    list_xy.append(dist_xy)
    if dist_xy <= 30:
        thr30 += 1
    if dist_xy <= 60:
        thr60 += 1
    if dist_xy <= 90:
        thr90 += 1
    '''

    # AUC
    fpr, tpr, thresholds = roc_curve(gt_flat, pred_flat)
    roc_auc = auc(fpr, tpr)
    auc_sum += roc_auc

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('data/test/pred/roc/' + str(i).zfill(6) + '.png')
   
print("size: ", size)
x /= size
y /= size
xy /= size
thr30 /= size
thr60 /= size
thr90 /= size
auc_sum /= len(gt_paths)
max_x = max(list_x)
min_x = min(list_x)
max_y = max(list_y)
min_y = min(list_y)
max_xy = max(list_xy)
min_xy = min(list_xy)
print(f"x: max {max_x}, min {min_x}, ave {x}")
print(f"y: max {max_y}, min {min_y}, ave {y}")
print(f"xy: max {max_xy}, min {min_xy}, ave {xy}")
print("auc: ", auc_sum)
print("Thr=30: ", thr30)
print("Thr=60: ", thr60)
print("Thr=90: ", thr90)
