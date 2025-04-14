import cv2
import numpy as np

import glob
import json
import os


def omni_img_to_3d_sphere(H, W):
    w = W // 2
    h = H // 2

    # make (1920, 3840) grid
    theta = np.linspace(-np.pi, np.pi, W)
    phi = np.linspace(np.pi/2, -np.pi/2, H)
    theta, phi = np.meshgrid(theta, phi)

    # 3d coordinate
    x = np.cos(phi) * np.cos(theta)
    y = np.cos(phi) * np.sin(theta)
    z = np.sin(phi)

    return x, y, z

def rotate_3d(x, y, z, roll, pitch, yaw):
    # rolling
    roll = roll * np.pi / 180
    pitch = pitch * np.pi / 180
    yaw = yaw * np.pi / 180

    # 3d rollong array
    mtx1 = np.array([[1, 0, 0],
                     [0, np.cos(roll), np.sin(roll)],
                     [0, -np.sin(roll), np.cos(roll)]])
    mtx2 = np.array([[np.cos(pitch), 0, -np.sin(pitch)],
                     [0, 1, 0],
                     [np.sin(pitch), 0, np.cos(pitch)]])
    mtx3 = np.array([[np.cos(yaw), np.sin(yaw), 0],
                     [-np.sin(yaw), np.cos(yaw), 0],
                     [0, 0, 1]])

    # inner product of rolling array
    mtx4 = np.dot(mtx3, np.dot(mtx2, mtx1))

    # formula of coordinate
    xx = mtx4[0][0] * x + mtx4[0][1] * y + mtx4[0][2] * z
    yy = mtx4[1][0] * x + mtx4[1][1] * y + mtx4[1][2] * z
    zz = mtx4[2][0] * x + mtx4[2][1] * y + mtx4[2][2] * z

    return xx, yy, zz

def remap_equirectangular(img_path, xx, yy, zz, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP):
    img = cv2.imread(img_path)
    H, W, C = img.shape
    w = W // 2
    h = H // 2
    r = h

    # transport to latitude and longitude
    phi = np.arcsin(zz) / (np.pi / 2)
    theta = np.arctan2(yy, xx) / np.pi

    # origin is center of img
    X = theta * w
    Y = phi * h

    # origin is left upper
    x = X + w
    y = -Y + h

    out = cv2.remap(img, x.astype(np.float32), y.astype(np.float32), interpolation, borderMode)

    return out

img_paths = glob.glob("data/test/frames/*/*.png")
for img_path in img_paths:
    img = cv2.imread(img_path)
    H, W, C = img.shape
    x, y, z = omni_img_to_3d_sphere(H, W)
    xx, yy, zz = rotate_3d(x, y, z, 0, 0, 0)
    out = remap_equirectangular(img_path, xx, yy, zz)
    # cv2.imwrite("./fisheye/000003.png", out)
