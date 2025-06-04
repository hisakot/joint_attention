import cv2
import numpy as np

import glob
import os

gt_paths = glob.glob("data/test/gt_heatmap_1ch/*/*.png")
gt_paths.sort()
pred_paths = glob.glob("data/test/pred/gaze_mult_selected_augmentation_no_rotate/*.png")
pred_paths.sort()
img_paths = glob.glob("data/test/frames/*/*.png")
img_paths.sort()
gazecone_paths = glob.glob("data/test/gazecone_mult/*/*.png")
gazecone_paths.sort()

H = 480
W = 960

for i, img_path in enumerate(img_paths):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (W, H))
    gazecone = cv2.imread(gazecone_paths[i], 0)
    gazecone = cv2.resize(gazecone, (W, H))
    gazecone = gazecone[:, :, np.newaxis]
    gt = cv2.imread(gt_paths[i], 0)
    gt = cv2.resize(gt, (W, H))
    gt = gt[:, :, np.newaxis]
    pred = cv2.imread(pred_paths[i], 0)
    pred = cv2.resize(pred, (W, H))
    pred = pred[:, :, np.newaxis]

    zeros = np.zeros((H, W, 1), dtype=np.uint8)
    result = np.concatenate([pred, zeros, gt], axis=2)
    '''
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    cv2.imwrite("data/test/pred/result/" + str(i).zfill(6) + ".png", result)

    # with RGB image

    _, gt_binary = cv2.threshold(gt, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    gt_contours, _ = cv2.findContours(gt_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gt_centers = []
    for cnt in gt_contours:
        area = cv2.contourArea(cnt)
        if area < 10:
            continue
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            gt_centers.append((cx, cy))

    _, pred_binary = cv2.threshold(pred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pred_contours, _ = cv2.findContours(pred_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_centers = []
    for cnt in pred_contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            pred_centers.append((cx, cy))

    result2 = cv2.addWeighted(img, 0.7, result, 1, 0)
    gazecone = np.concatenate([zeros, gazecone, gazecone], axis=2)
    for gt_center in gt_centers:
        result2 = cv2.drawMarker(result2, gt_center, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3)
    for pred_center in pred_centers:
        result2 = cv2.drawMarker(result2, pred_center, (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3)
    result3 = cv2.addWeighted(result2, 0.7, gazecone, 1, 0)

    '''
    cv2.imshow("result2", result2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    cv2.imwrite("data/test/pred/result2/" + str(i).zfill(6) + ".png", result2)
    cv2.imwrite("data/test/pred/result3/" + str(i).zfill(6) + ".png", result3)
