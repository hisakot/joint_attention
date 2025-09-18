import argparse
import glob
import math
import os
import time
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from pytorch_msssim import ssim

import config
import dataset
import PJAE_conv
import transGan
import swin_unet

def print_memory_usage():
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(F"GPU Memory Allocated: {allocated:.2f} MB")
    print(F"GPU Memory Reserved: {reserved:.2f} MB")

def train(train_dataloader, net, loss_functions, optimizer, device):
    net.train()

    total_loss =  0
    cos_total, mse_total, mae_total, kl_total, ssim_total = 0, 0, 0, 0, 0
    start_time = time.time()

    num_batches = len(train_dataloader)

    with tqdm(total=len(train_dataloader)) as pbar:
        try:
            for data in train_dataloader:
                inputs = data[0].to(device)
                '''
                for key, val in inp.items():
                    if torch.is_tensor(val):
                        inp[key] = val.to(device)
                '''

                targets = data[1].to(device)
                if inputs is None or targets is None:
                    continue

                pred = net(inputs)

                loss = 0
                cos_loss, mse_loss, mae_loss, kl_loss, ssim_loss = 0, 0, 0, 0, 0
                for loss_function in loss_functions:
                    if loss_function == "cos_similarity":
                        pred_flat = pred.view(pred.size(0), -1)
                        targets_flat = targets.view(targets.size(0), -1)
                        cos_loss = (1 - F.cosine_similarity(pred_flat, targets_flat)).mean()
                        loss += cos_loss
                        cos_total += cos_loss
                    elif loss_function == "MSE":
                        lossfunc = nn.MSELoss()
                        mse_loss = lossfunc(pred, targets)
                        loss += mse_loss
                        mse_total += mse_loss
                    elif loss_function == "MAE":
                        lossfunc = nn.L1Loss()
                        mae_loss = lossfunc(pred, targets)
                        loss += mae_loss
                        mze_total += mae_loss
                    elif loss_function == "KLDiv":
                        lossfunc = nn.KLDivLoss(reduction='batchmean')
                        # pred
                        pred_flat = pred.view(pred.size(0), -1)
                        log_pred = F.log_softmax(pred_flat, dim=1)
                        # targets
                        targets_flat = targets.view(targets.size(0), -1)
                        targets_sum = targets_flat.sum(dim=1, keepdim=True)
                        targets_norm = torch.where(targets_sum > 0, targets_flat / targets_sum, targets_flat)
                        kl_loss = lossfunc(log_pred, targets_norm)
                        loss += kl_loss
                        kl_total += kl_loss
                    elif loss_function == "combined_loss":
                        loss += compute_all_losses(pred, targets)
                    elif loss_function == "SSIM":
                        ssim_loss = 1 - ssim(pred, targets, data_range=1, size_average=True)
                        loss += ssim_loss
                        ssim_total += ssim_loss

                optimizer.zero_grad()
                loss.backward()
                '''
                torch.nn.utils.clip_grad_norm_(swin_unet.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(swin_t.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(fuse.parameters(), 0.5)
                # torch.nn.utils.clip_grad_norm_(resnet50.parameters(), 0.5)
                '''
                optimizer.step()

                total_loss += loss.item()
                pbar.update()
        except TypeError:
            print("Error: TypeError")
            pass

# return total_loss / len(train_dataloader)
    return [total_loss / len(train_dataloader), cos_total.item(), kl_total.item(), ssim_total.item()]

def evaluate(val_dataloader, net, loss_functions, device):
    '''
    resnet50.eval()
    unet.eval()
    fuse.eval()
    swin_t.eval()
    swin_unet.eval()
    spatiotemporal.eval()
    '''
    net.eval()
    total_loss = 0
    cos_total, mse_total, mae_total, kl_total, ssim_total = 0, 0, 0, 0, 0

    with torch.no_grad():
        with tqdm(total=len(val_dataloader)) as pbar:
            try:
                for data in val_dataloader:
                    inputs = data[0].to(device)
                    '''
                    for key, val in inp.items():
                        if torch.is_tensor(val):
                            inp[key] = val.to(device)
                    '''

                    targets = data[1].to(device)
                    if inputs is None or targets is None:
                        continue

                    pred = net(inputs)

                    loss = 0
                    cos_loss, mse_loss, mae_loss, kl_loss, ssim_loss = 0, 0, 0, 0, 0
                    for loss_function in loss_functions:
                        if loss_function == "cos_similarity":
                            pred_flat = pred.view(pred.size(0), -1)
                            targets_flat = targets.view(targets.size(0), -1)
                            cos_loss = (1 - F.cosine_similarity(pred_flat, targets_flat)).mean()
                            loss += cos_loss
                            cos_total += cos_loss
                        elif loss_function == "MSE":
                            lossfunc = nn.MSELoss()
                            mse_loss = lossfunc(pred, targets)
                            loss += mse_loss
                            mse_total += mse_loss
                        elif loss_function == "MAE":
                            lossfunc = nn.L1Loss()
                            mae_loss = lossfunc(pred, targets)
                            loss += mae_loss
                            mae_total += mae_loss
                        elif loss_function == "KLDiv":
                            lossfunc = nn.KLDivLoss(reduction='batchmean')
                            # pred
                            pred_flat = pred.view(pred.size(0), -1)
                            log_pred = F.log_softmax(pred_flat, dim=1)
                            # targets
                            targets_flat = targets.view(targets.size(0), -1)
                            targets_sum = targets_flat.sum(dim=1, keepdim=True)
                            targets_norm = torch.where(targets_sum > 0, targets_flat / targets_sum, targets_flat)
                            kl_loss = lossfunc(log_pred, targets_norm)
                            loss += kl_loss
                            kl_total += kl_loss
                        elif loss_function == "combined_loss":
                            loss = compute_all_losses(pred, targets)
                        elif loss_function == "SSIM":
                            ssim_loss = 1 - ssim(pred, targets, data_range=1, size_average=True)
                            loss += ssim_loss
                            ssim_total += ssim_loss

                    total_loss += loss.item()
                    pbar.update()
            except TypeError:
                print("Error: TypeError")
                pass

# return total_loss / len(val_dataloader)
    return [total_loss / len(val_dataloader), cos_total.item(), kl_total.item(), ssim_total.item()]

def collate_fn(batch):
    inputs, labels = zip(*batch)

    length = torch.tensor([x.shape[0] for x in inputs])

    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    attention_mask = (padded_inputs != 0).float()

    labels = torch.stack(labels)
    return padded_inputs, attention_mask, labels, length

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
    # B = pred_logits.shape[0]
    B, C, H, W = pred_logits.shape

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

def main():
    torch.manual_seed(0)

    parser = argparse.ArgumentParser(description="Process some integers")
    parser.add_argument("--batch_size", required=False, default=1, type=int)
    parser.add_argument("--checkpoint", required=False,
                        help="if you want to retry training, write model path")
    args = parser.parse_args()
    batch_size = args.batch_size

    cfg = config.Config()
    lr = cfg.lr
    seq_len = cfg.seq_len

    img_height = cfg.img_height
    img_width = cfg.img_width

    '''
    net = PJAE_conv.ModelSpatial(in_ch=5)
    net = transGan.TransGAN_LSTM(patch_size=10, emb_size=512, num_heads=2, forward_expansion=4,
                                 img_height=img_height, img_width=img_width, in_ch=5, seq_len=seq_len)
    '''
    net = swin_unet.SwinUnet(img_height=img_height, img_width=img_width,
                             patch_size=2, in_chans=5, num_classes=1, embed_dim=48,
                             lstm_input_dim=384, lstm_hidden_dim=384, seq_len=seq_len)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        print("---------- Use GPU ----------")
        # net = nn.DataParallel(net)
    else:
        print("---------- Use CPU ----------")
    net.to(device)

    # loss_function = nn.CrossEntropyLoss()
    # loss_functions = ["MSE", "MAE", "cos_similarity", "KLDiv", "combined_loss", "SSIM"]
    loss_functions = ["cos_similarity", "KLDiv", "SSIM"]
    optimizer = optim.SGD(net.parameters(), lr=lr)
    # optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=2e-2)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=1)

    writer = SummaryWriter(log_dir="logs")

    num_cpu = os.cpu_count()
    num_cpu = num_cpu // 4
    print("number of cpu: ", num_cpu)

    train_loss_list = list()
    val_loss_list = list()

    train_data_dir = "data/ue/train"
    val_data_dir = "data/ue/val"
    train_data = dataset.Dataset(train_data_dir,
                                 img_height=img_height, img_width=img_width,
                                 seq_len=seq_len, transform=None, is_train=True)
    val_data = dataset.Dataset(val_data_dir,
                               img_height=img_height, img_width=img_width,
                               seq_len=seq_len, transform=None, is_train=False)

    train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                  shuffle=False, num_workers=num_cpu) # FIXME shuffle
    val_dataloader = DataLoader(val_data, batch_size=batch_size,
                                shuffle=False, num_workers=num_cpu)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        net.load_state_dict(checkpoint["net_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        train_loss_list = checkpoint["train_loss_list"]
        val_loss_list = checkpoint["val_loss_list"]
        for i, train_loss in enumerate(train_loss_list):
            writer.add_scalar("Train Loss", train_loss, i+1)
        for i, val_loss in enumerate(val_loss_list):
            writer.add_scalar("Validation Loss", val_loss, i+1)
        print("Reload midel : ", start_epoch, "and restart training")
    else:
        start_epoch = 0

    epochs = 1000
    early_stopping = [np.inf, 50, 0]
    for epoch in range(epochs):
        epoch += start_epoch
        print(f"--------------------\nEpoch {epoch+1}")
        print("early stopping: ", early_stopping)
        print(f"lr: {scheduler.get_last_lr()[0]}")
        try:
            # train
            train_loss = train(train_dataloader, net,
                               loss_functions, optimizer, device)
            train_loss_list.append(train_loss)

            # test
            with torch.no_grad():
                val_loss = evaluate(val_dataloader, net,
                                    loss_functions, device)
                val_loss_list.append(val_loss)

            print("Epoch %d : train_loss %.3f" % (epoch + 1, train_loss[0]))
            print("Epoch %d : val_loss %.3f" % (epoch + 1, val_loss[0]))
            print(train_loss, val_loss)

            # lr_scheduler
            scheduler.step()

            # save model
            if val_loss[0] < early_stopping[0]:
                early_stopping[0] = val_loss[0]
                early_stopping[2] = 0
                torch.save({"epoch" : epoch + 1,
                            "net_state_dict" : net.state_dict(),
                            "optimizer_state_dict" : optimizer.state_dict(),
                            "train_loss_list" : train_loss_list,
                            "val_loss_list" : val_loss_list,
                            }, "save_models/lstm_trial.pth")
            else:
                early_stopping[2] += 1
                if early_stopping[2] == early_stopping[1]:
                    break

            # tensorboard
            writer.add_scalar("Train Loss", train_loss[0], epoch + 1)
            writer.add_scalar("Train cosLoss", train_loss[1], epoch + 1)
            writer.add_scalar("Train klLoss", train_loss[2], epoch + 1)
            writer.add_scalar("Train ssimLoss", train_loss[3], epoch + 1)
            writer.add_scalar("Valid Loss", val_loss[0], epoch + 1)
            writer.add_scalar("Valid cosLoss", val_loss[1], epoch + 1)
            writer.add_scalar("Valid klLoss", val_loss[2], epoch + 1)
            writer.add_scalar("Valid ssimLoss", val_loss[3], epoch + 1)
            print("log updated")

        except ValueError:
            continue

if __name__ == "__main__":
    main()
