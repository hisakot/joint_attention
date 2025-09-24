import glob
import math
import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

matplotlib.use('Agg')

# gt_paths = glob.glob("data/test/gt_heatmap_1ch_large/*/*.png")
gt_paths = glob.glob("data/ue/test/gt_heatmap_1ch_large/*/*.png")
gt_paths.sort()
pred_paths = glob.glob("data/test/pred/result_hm/*.png")
# pred_paths = glob.glob("data/test/pred/transGANv2_lr1e4_gazearea/*.png")
# pred_paths = glob.glob("data/test/pred/swinunet_no_skipconnection/*.png")
# pred_paths = glob.glob("data/test/pred/result1/*.png")
pred_paths.sort()
# img_paths = glob.glob("data/test/frames/*/*.png")
img_paths = glob.glob("data/ue/test/frames/*/*.png")
img_paths.sort()

H = 640
W = 1280

x, y, xy, auc_sum = 0, 0, 0, 0
list_x = []
list_y = []
list_xy = []
thr30 = 0
thr60 = 0
thr90 = 0
size = 0
for i, pred_path in tqdm(enumerate(pred_paths), total=len(pred_paths)):
    # GT
    gt = cv2.imread(gt_paths[i], 0)
    gt = cv2.resize(gt, (W, H))
    gt_float = gt.astype(np.float64)
    gt_float /= 255.
    gt_argmax = list(zip(*np.where(gt_float == np.max(gt_float))))
    # gt_argmax = np.unravel_index(np.argmax(gt_float), gt_float.shape)
    gt_flat = gt_float.reshape(-1)
    gt_flat = np.where(gt_flat > 0, 1, 0)

    # PRED
    pred = cv2.imread(pred_path, 0)
    pred = cv2.resize(pred, (W, H))
    pred_float = pred.astype(np.float64)
    pred_float /= 255.
    pred_argmax = np.unravel_index(np.argmax(pred_float), pred_float.shape)
    pred_flat = pred_float.reshape(-1)

    # IoU
    _, gt_iou = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
    gt_iou_flat = gt_iou.reshape(-1)
    gt_iou_flat = np.where(gt_iou_flat > 127, 255, 0)

    thr = int(np.max(pred) / 2)
    _, pred_iou = cv2.threshold(pred, thr, 255, cv2.THRESH_BINARY)
    pred_iou_flat = pred_iou.reshape(-1)
    pred_iou_flat = np.where(pred_iou_flat > 127, 255, 0)

    gt_pix, pred_pix, both_pix = 0, 0, 0
    for j in range(gt_iou_flat.shape[0]):
        if gt_iou_flat[j] != 0:
            gt_pix += 1
        if pred_iou_flat[j] != 0:
            pred_pix += 1
        if gt_iou_flat[j] != 0 and pred_iou_flat[j] != 0:
            both_pix += 1
    iou = both_pix / (gt_pix + pred_pix - both_pix)


    # concat image
    zeros = np.zeros((H, W, 1), dtype=np.uint8)
    concat = np.concatenate([pred[:, :, np.newaxis], zeros, gt[:, :, np.newaxis]], axis=2)
    img = cv2.imread(img_paths[i])
    img = cv2.resize(img, (W, H))
    result = cv2.addWeighted(img, 1, concat, 1, 0)

    # if using pred moment
    _, gt_binary = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
    gt_contours, _ = cv2.findContours(gt_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for gt_cnt in gt_contours:
        gt_M = cv2.moments(gt_cnt)
        if gt_M['m00'] != 0:
            gt_x = int(gt_M['m10'] / gt_M['m00'])
            gt_y = int(gt_M['m01'] / gt_M['m00'])
        else:
            continue

        '''
        pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
        cv2.imshow("jet", pred)
        cv2.waitKey(0)
        '''
        # unique, count = np.unique(pred, return_counts=True)
        thr = int(np.max(pred) / 2)
        _, pred_binary = cv2.threshold(pred, thr, 255, cv2.THRESH_BINARY)
        # pred_binary = cv2.cvtColor(pred_binary, cv2.COLOR_BGR2GRAY)
        pred_contours, _ = cv2.findContours(pred_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_brightness = -1
        best_contour = None

        dist_min = [0, 0, math.sqrt(W**2 + H**2)]
        for cnt in pred_contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            mask = np.zeros_like(pred, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)
            mean_val = cv2.mean(pred, mask=mask)[0]
            if mean_val > max_brightness:
                max_brightness = mean_val
                best_contour = cnt
            cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)

        cv2.drawContours(result, [best_contour], -1, (0, 255, 0), 2)
        M = cv2.moments(best_contour)
        if M['m00'] != 0:
            size += 1
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            dist_x = abs(cx - gt_x)
            if dist_x > W / 2:
                dist_x = W - dist_x
            dist_y = abs(cy - gt_y)
            dist_xy = math.sqrt(dist_x**2 + dist_y**2)
            if dist_min[2] > dist_xy:
                dist_min[0] = dist_x
                dist_min[1] = dist_y
                dist_min[2] = dist_xy
            cv2.line(result, (cx, cy), (gt_x, gt_y), color=(255, 255, 255),
                     thickness=2, lineType=cv2.LINE_AA)
            cv2.drawMarker(result, (cx, cy), color=(255, 0, 0), 
                           markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3)
            cv2.drawMarker(result, (gt_x, gt_y), color=(0, 0, 255), 
                           markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3)

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
    dist_xy = math.sqrt(dist_x**2 + dist_y**2)
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

    cv2.imwrite('data/test/pred/result_color/' + str(i).zfill(6) + '.png', result)
   
print("size: ", size)
x /= size
y /= size
xy /= size
thr30 /= size
thr60 /= size
thr90 /= size
auc_sum /= len(pred_paths)
max_x = max(list_x)
min_x = min(list_x)
std_x = np.std(list_x)
max_y = max(list_y)
min_y = min(list_y)
std_y = np.std(list_y)
max_xy = max(list_xy)
min_xy = min(list_xy)
std_xy = np.std(list_xy)
print(f"x: max {max_x}, min {min_x}, ave {x}, std {std_x}")
print(f"y: max {max_y}, min {min_y}, ave {y}, std {std_y}")
print(f"xy: max {max_xy}, min {min_xy}, ave {xy}, std {std_xy}")
print("auc: ", auc_sum)
print("Thr=30: ", thr30)
print("Thr=60: ", thr60)
print("Thr=90: ", thr90)
print("IoU: ", iou)
