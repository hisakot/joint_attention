import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize_prob(heatmap, eps=1e-12):
    if heatmap.dim() == 4:
        heatmap = heatmap.squeeze(1)
    heatmap_sum = heatmap.sum(dim=(-2, -1), keepdim=True)
    return heatmap / (heatmap_sum + eps)

def soft_argmax_2d(prob):
    B, H, W = prob.shape
    xs = torch.linspace(0, W - 1, W, device=prob.device)
    ys = torch.linspace(0, H - 1, H, device=prob.device)
    xs_grid, ys_grid = torch.meshgrid(xs, ys, indexing="xy")
    xs_grid = xs_grid.unsqueeze(0).expand(B, -1, -1)
    ys_grid = ys_grid.unsqueeze(0).expand(B, -1, -1)
    x = (prob * xs_grid).sum(dim=(1, 2))
    y = (prob * ys_grid).sum(dim=(1, 2))
    return x, y

def circular_moments_normalized(prob):
    # 水平方向の平均と円周分散 (B,H,W) -> mean_x, circ_var_x
    B, H, W = prob.shape
    px = prob.sum(dim=1)  # (B,W)
    idx_x = torch.arange(W, device=prob.device).view(1, W)
    theta_x = 2 * math.pi * idx_x / W
    C = (px * torch.cos(theta_x)).sum(dim=1)
    S = (px * torch.sin(theta_x)).sum(dim=1)
    R = torch.sqrt(C**2 + S**2).clamp_min(1e-12)
    circ_var = 1.0 - R
    mean_theta = torch.atan2(S, C) % (2*math.pi)
    mean_x = (mean_theta / (2*math.pi)) * W
    return mean_x, circ_var

def variance_y_normalized(prob):
    # 垂直方向の分散 (B,H,W) -> mean_y, var_y
    B, H, W = prob.shape
    ys = torch.linspace(0, H-1, H, device=prob.device).view(1, H, 1)
    mean_y = (prob * ys).sum(dim=(1,2), keepdim=True)
    var_y = (prob * (ys - mean_y)**2).sum(dim=(1,2)).squeeze() / (H**2)  # 無次元化
    return mean_y.squeeze(), var_y

# spherical surface distance
def lonlat_from_xy(x, y, W, H):
    # x: [0, W), y: [0, H) -> longitude(lambda): [-pi, pi), latitude(phi): [-pi/2, pi/2]
    lam = (x / W) * 2 * math.pi - math.pi
    phi = (0.5 - y / H) * math.pi
    return lam, phi

def sph2vec(lam, phi):
    x = torch.cos(phi) * torch.cos(lam)
    y = torch.cos(phi) * torch.sin(lam)
    z = torch.sin(phi)
    return torch.stack([x, y, z], dim=-1) # (B, 3)

def compute_all_losses(pred_logits, gt_heatmap, tau=1.0, lambda_kl=1.0,
                       lambda_coord=1.0, lambda_variance=1.0, lambda_ang=1.0):
    """
    pred_logits: (B,1,H,W) - ネットワーク出力 (softmax前)
    gt_heatmap:  (B,1,H,W) - ガウシアンラベル (非正規化)
    """
    B = pred_logits.shape[0]
    print(pred_logits.shape)

    # Normalize
    pred_prob = F.softmax(pred_logits.view(B, -1) / tau, dim=-1).view(B,1,H,W).squeeze(1)
    gt_prob   = normalize_prob(gt_heatmap)

    # KL Divergence
    loss_kl = F.kl_div(pred_prob.log().view(B,-1),
                       gt_prob.view(B,-1),
                       reduction="batchmean")

    # Coordinate
    x_pred, y_pred = soft_argmax_2d(pred_prob)
    x_gt, y_gt = soft_argmax_2d(gt_prob)
    dx = torch.abs(x_pred - x_gt)
    dx = torch.min(dx, torch.tensor(W, device=dx.device) - dx)  # wrap-around
    dy = torch.abs(y_pred - y_gt)
    loss_coord = (((dx/W)**2 + (dy/H)**2).mean())

    # Variance
    _, vx_p = circular_moments_normalized(pred_prob)
    _, vx_g = circular_moments_normalized(gt_prob)
    _, vy_p = variance_y_normalized(pred_prob)
    _, vy_g = variance_y_normalized(gt_prob)
    eps = 1e-6
    loss_var_x = (torch.log(vx_p + eps) - torch.log(vx_g + eps)).abs().mean()
    loss_var_y = (torch.log(vy_p + eps) - torch.log(vy_g + eps)).abs().mean()
    loss_variance = 0.5 * (loss_var_x + loss_var_y)

    # Angle
    lam_pred, phi_pred = lonlat_from_xy(x_pred, y_pred, W, H)
    lam_gt, phi_gt = lonlat_from_xy(x_gt, y_gt, W, H)
    v_pred = sph2vec(lam_pred, phi_pred)
    v_gt = sph2vec(lam_gt, phi_gt)
    cosang = (v_pred * v_gt).sum(dim=-1).clamp(-1.0, 1.0)
    ang = torch.acos(cosang) # [0, pi]
    loss_ang = ang.mean()

    '''
    angle_pred = torch.atan2(y_pred - H/2, x_pred - W/2)
    angle_gt   = torch.atan2(y_gt - H/2, x_gt - W/2)
    loss_ang   = (angle_pred - angle_gt).abs().mean()  # 単純L1
    '''

    loss = lambda_kl * loss_kl + lambda_coord * loss_coord + lambda_variance * loss_variance + lambda_ang * loss_ang

    return loss


W = 1920
H = 960
gt = cv2.imread("data/test/gt_heatmap_1ch/ds_005/000000.png")
gt = cv2.resize(gt, (W, H))
gt = gt.astype(np.float32)
gt /= 255.
gt = np.transpose(gt, (2, 0, 1)) # C, H, W
gt = gt[np.newaxis, 0:1, :, :]
gt = torch.tensor(gt, dtype=torch.float32)

pred = cv2.imread("data/test/pred/result1/000000.png")
pred = cv2.resize(pred, (W, H))
pred = pred.astype(np.float32)
pred /= 255.
pred = np.transpose(pred, (2, 0, 1)) # C, H, W
pred = pred[np.newaxis, 0:1, :, :]
pred = torch.tensor(pred, dtype=torch.float32)

a, b, c, d = compute_all_losses(pred, gt, W, H)
