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
pred_paths = glob.glob("data/test/pred/gazecone_close_kpt_lined_cossim/*.png")
pred_paths.sort()
print(len(gt_paths), len(pred_paths))

x, y, xy, auc = 0, 0, 0, 0
for i, gt_path in tqdm(enumerate(gt_paths), total=len(gt_paths)):
    gt = cv2.imread(gt_path, 0)
    gt = cv2.resize(gt, (640, 320))
    gt = gt.astype(np.float64)
    gt /= 255.
    gt_max = np.max(gt)
    gt_argmax = np.unravel_index(np.argmax(gt), gt.shape)
    gt_flat = gt.reshape(-1)
    gt_flat = np.where(gt_flat > 0, 1, 0)

    pred_path = pred_paths[i]
    pred = cv2.imread(pred_path, 0)
    pred = cv2.resize(pred, (640, 320))
    pred = pred.astype(np.float64)
    pred /= 255.
    pred_max = np.max(pred)
    pred_argmax = np.unravel_index(np.argmax(pred), pred.shape)
    pred_flat = pred.reshape(-1)

    dist_x = abs(pred_argmax[0] - gt_argmax[0])
    dist_y = abs(pred_argmax[1] - gt_argmax[1])
    dist_xy = math.sqrt(dist_x^2 + dist_y^2)
    x += dist_x
    y += dist_y
    xy += dist_xy

    fpr, tpr, thresholds = roc_curve(gt_flat, pred_flat)
    roc_auc = auc(fpr, tpr)
    auc += roc_auc

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('data/test/pred/roc/roc_example' + str(i).zfill(6) + '.png')

    print(dist_x, dist_y, dist_xy, roc_auc)

x /= len(gt_paths)
y /= len(gt_paths)
xy /= len(gt_paths)
auc /= len(gt_paths)
print(x, y, xy, auc)

