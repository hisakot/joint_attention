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

import config
import dataset
import swin_unet

def print_memory_usage():
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(F"GPU Memory Allocated: {allocated:.2f} MB")
    print(F"GPU Memory Reserved: {reserved:.2f} MB")

def train(train_dataloader, net, loss_function, optimizer, device):
    net.train()

    total_loss =  0
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

                if loss_function == "cos_similarity":
                    pred = pred.view(pred.size(0), -1)
                    targets = targets.view(targets.size(0), -1)
                    cos_loss = F.cosine_similarity(pred, targets)
                    loss = (1 - cos_loss).mean()
                elif loss_function == "MSE":
                    lossfunc = nn.MSELoss()
                    loss = lossfunc(pred, targets)
                elif loss_function == "MAE":
                    lossfunc = nn.L1Loss()
                    loss = lossfunc(pred, targets)
                elif loss_function == "KLDiv":
                    lossfunc = nn.KLDivLoss(reduction='batchmean')
                    # pred
                    pred = pred.view(pred.size(0), -1)
                    log_pred = F.log_softmax(pred, dim=1)
                    # targets
                    targets = targets.view(targets.size(0), -1)
                    targets_sum = targets.sum(dim=1, keepdim=True)
                    targets_norm = torch.where(targets_sum > 0, targets / targets_sum, targets)
                    loss = lossfunc(log_pred, targets_norm)
                elif loss_function == "combined_loss":
                    loss = combined_gaze_loss(pred, targets)

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

    return total_loss / len(train_dataloader)

def evaluate(val_dataloader, net, loss_function, device):
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

                    if loss_function == "cos_similarity":
                        pred = pred.view(pred.size(0), -1)
                        targets = targets.view(targets.size(0), -1)
                        cos_loss = F.cosine_similarity(pred, targets)
                        loss = (1 - cos_loss).mean()
                    elif loss_function == "MSE":
                        lossfunc = nn.MSELoss()
                        loss = lossfunc(pred, targets)
                    elif loss_function == "MAE":
                        lossfunc = nn.L1Loss()
                        loss = lossfunc(pred, targets)
                    elif loss_function == "KLDiv":
                        lossfunc = nn.KLDivLoss(reduction='batchmean')
                        # pred
                        pred = pred.view(pred.size(0), -1)
                        log_pred = F.log_softmax(pred, dim=1)
                        # targets
                        targets = targets.view(targets.size(0), -1)
                        targets_sum = targets.sum(dim=1, keepdim=True)
                        targets_norm = torch.where(targets_sum > 0, targets / targets_sum, targets)
                        loss = lossfunc(log_pred, targets_norm)
                    elif loss_function == "combined_loss":
                        loss = combined_gaze_loss(pred, targets)

                    total_loss += loss.item()
                    pbar.update()
            except TypeError:
                print("Error: TypeError")
                pass

    return total_loss / len(val_dataloader)

def collate_fn(batch):
    inputs, labels = zip(*batch)

    length = torch.tensor([x.shape[0] for x in inputs])

    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    attention_mask = (padded_inputs != 0).float()

    labels = torch.stack(labels)
    return padded_inputs, attention_mask, labels, length

def normalize_prob(heatmap, eps=1e-12):
    B, C, H, W = heatmap.shape
    heatmap = heatmap.view(B, -1)
    heatmap = heatmap / (heatmap.sum(dim=-1, keepdim=True) + eps)
    return heatmap.view(B, C, H, W)

def log_prob_from_logits(logits):
    B, C, H, W = logits.shape
    logits = logits.view(B, -1)
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return logits.view(B, C, H, W)

# KL
def kl_div_loss(pred_logits, gt_prob, reduce=True):
    log_p = log_prob_from_logits(pred_logits)
    p = log_p.exp()
    # KL(gt || p)
    kl = (gt_prob * (gt_prob.clamp_min(1e-12).log() - log_p)).sum(dim=(1, 2, 3))
    return kl.mean() if reduce else kl

def kl(pred, gt):
    lossfunc = nn.KLDivLoss(reduction='batchmean')
    # pred
    pred = pred.view(pred.size(0), -1)
    log_pred = F.log_softmax(pred, dim=1)
    # targets
    targets = gt.view(gt.size(0), -1)
    targets_sum = targets.sum(dim=1, keepdim=True)
    targets_norm = torch.where(targets_sum > 0, targets / targets_sum, targets)
    loss = lossfunc(log_pred, targets_norm)
    return loss

# soft-argmax + wrap-around distance
def soft_argmax_2d(logits, tau=1.0):
    # smaller tau -> sharper

    B, C, H, W = logits.shape
    flat = (logits / tau).view(B, -1)
    prob = F.softmax(flat, dim=-1).view(B, H, W)

    xs = torch.linspace(0, W - 1, W, device=logits.device)
    ys = torch.linspace(0, H - 1, H, device=logits.device)
    xs, ys = torch.meshgrid(xs, ys, indexing="xy")
    x = (prob * xs).sum(dim=(1, 2))
    y = (prob * ys).sum(dim=(1, 2))
    return x, y, prob

def circular_dx(pred_x, gt_x, W):
    dx = torch.abs(pred_x - gt_x)
    return torch.min(dx, W - dx)

def coord_loss_from_logits(pred_logits, gt_prob, W, H, tau=1.0):
    xh, yh, _ = soft_argmax_2d(pred_logits, tau)

    B, C, H, W = gt_prob.shape
    xs = torch.linspace(0, W - 1, W, device=gt_prob.device)
    ys = torch.linspace(0, H - 1, H, device=gt_prob.device)
    xs, ys = torch.meshgrid(xs, ys, indexing="xy")
    xg = (gt_prob[:, 0] * xs).sum(dim=(1, 2))
    yg = (gt_prob[:, 0] * ys).sum(dim=(1, 2))

    dx = circular_dx(xh, xg, W)
    dy = torch.abs(yh - yg)
    return (dx**2 + dy**2).mean()

# Variance
def circular_moments(prob, axis_len, axis='x'):
    B, H, W = prob.shape
    if axis == 'x':
        px = prob.sum(dim=1)
        idx = torch.arange(W, device=prob.device)
        theta = 2 * math.pi * idx / W
    else:
        py = prob.sum(dim=2)
        idx = torch.arange(H, device=prob.device)
        theta = 2 * math.pi * idx / H

    ct = torch.cos(theta)[None, :] # (1, L)
    st = torch.sin(theta)[None, :]

    if axis == 'x':
        C = (px * ct).sum(dim=1)
        S = (px * st).sum(dim=1)
        R = torch.sqrt(C**2 + S**2).clamp_min(1e-12)
        mean_theta = torch.atan2(S, C) % (2 * math.pi)
        circ_var = 1 - R # 0=forcusing 1=dispersing
        mean_idx = (mean_theta / (2 * math.pi)) * W
    else:
        C = (py * ct).sum(dim=1)
        S = (py * st).sum(dim=1)
        R = torch.sqrt(C**2 + S**2).clamp_min(1e-12)
        mean_theta = torch.atan2(S, C) % (2 * math.pi)
        circ_var = 1 - R # 0=forcusing 1=dispersing
        mean_idx = (mean_theta / (2 * math.pi)) * W

    return mean_idx, circ_var

def variance_y(prob):
    B, H, W = prob.shape
    ys = torch.linspace(0, H - 1, H, device=prob.device)
    mean_y = (prob * ys[:, None]).sum(dim=(1, 2))
    var_y = (prob * (ys[:, None] - mean_y[:, None, None])**2).sum(dim=(1, 2))
    return mean_y, var_y

def varience_matching_loss(pred_logits, gt_prob, W, H, tau=1.0):
    _, _, p = soft_argmax_2d(pred_logits, tau)

    # x: circular variance
    mx_p, vx_p = circular_moments(p, W, axis='x')
    mx_g, vx_g = circular_moments(gt_prob[:, 0], W, axis='x')
    # y: variance
    my_p, vy_p = variance_y(p)
    my_g, vy_g = variance_y(gt_prob[:, 0])

    loss_var_x = (vx_p - vx_g).abs().mean()
    loss_var_y = (vy_p - vy_g).abs().mean()
    return (loss_var_x + loss_var_y) * 0.5

# spherical surface distance
def lonlat_from_xy(x, y, W, H):
    # x: [0, W), y: [0, H) -> longitude(lambda): [-pi, pi), latitude(phi): [-pi/2, pi/2]
    lam = (x / W) * 2 * math.pi - math.pi
    phi = (0.5 - y / H) * math.pi
    return lam, phi

def angular_distance_loss(pred_logits, gt_prob, W, H, tau=1.0):
    xh, yh, p = soft_argmax_2d(pred_logits, tau)
    B = xh.shape[0]

    # GT
    xs = torch.linspace(0, W - 1, W, device=gt_prob.device)
    ys = torch.linspace(0, H - 1, H, device=gt_prob.device)
    xs, ys = torch.meshgrid(xs, ys, indexing="xy")
    xg = (gt_prob[:, 0] * xs).sum(dim=(1, 2))
    yg = (gt_prob[:, 0] * ys).sum(dim=(1, 2))

    lam_h, phi_h = lonlat_from_xy(xh, yh, W, H)
    lam_g, phi_g = lonlat_from_xy(xg, yg, W, H)

    # spherical surface angular
    def sph2vec(lam, phi):
        x = torch.cos(phi) * torch.cos(lam)
        y = torch.cos(phi) * torch.sin(lam)
        z = torch.sin(phi)
        return torch.stack([x, y, z], dim=-1) # (B, 3)

    vh = sph2vec(lam_h, phi_h)
    vg = sph2vec(lam_g, phi_g)
    cosang = (vh * vg).sum(dim=-1).clamp(-1.0, 1.0)
    ang = torch.acos(cosang) # [0, pi]
    return ang.mean()

# loss integration
def combined_gaze_loss(
        pred_logits, gt_heatmap,
        lam_kl=1.0, lam_coord=0.0, lam_var=0.0, lam_ang=1.0, tau=0.8):
    B, _, H, W = pred_logits.shape
    gt_prob = normalize_prob(gt_heatmap)

    # loss_kl = kl_div_loss(pred_logits, gt_prob)
    loss_kl = kl(pred_logits, gt_prob)
    loss_coord = coord_loss_from_logits(pred_logits, gt_prob, W, H, tau)
    loss_var = varience_matching_loss(pred_logits, gt_prob, W, H, tau)
    if lam_ang > 0:
        loss_ang = angular_distance_loss(pred_logits, gt_prob, W, H, tau)
    else:
        loss_ang = pred_logits.new_tensor(0.0)

    total = lam_kl * loss_kl + lam_coord * loss_coord + lam_var * loss_var + lam_ang + loss_ang
    return total


def main():

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
    net = swin_unet.SwinTransformerSys(img_height=img_height, img_width=img_width,
                                       in_chans=5, num_classes=1, window_size=5,
                                       lstm_input_dim=768, lstm_hidden_dim=768,
                                       seq_len=seq_len)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        print("---------- Use GPU ----------")
        # net = nn.DataParallel(net)
    else:
        print("---------- Use CPU ----------")
    net.to(device)

    # loss_function = nn.CrossEntropyLoss()
    # loss_function = "MSE"
    # loss_function = "MAE"
    # loss_function = "cos_similarity"
    # loss_function = "KLDiv"
    loss_function = "combined_loss"
    optimizer = optim.SGD(net.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=1)

    writer = SummaryWriter(log_dir="logs")

    num_cpu = os.cpu_count()
    num_cpu = num_cpu // 4
    print("number of cpu: ", num_cpu)

    train_loss_list = list()
    val_loss_list = list()

    train_data_dir = "data/train"
    val_data_dir = "data/val"
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
                               loss_function, optimizer, device)
            train_loss_list.append(train_loss)

            # test
            with torch.no_grad():
                val_loss = evaluate(val_dataloader, net,
                                    loss_function, device)
                val_loss_list.append(val_loss)

            print("Epoch %d : train_loss %.3f" % (epoch + 1, train_loss))
            print("Epoch %d : val_loss %.3f" % (epoch + 1, val_loss))

            # lr_scheduler
            scheduler.step()

            # save model
            if val_loss < early_stopping[0]:
                early_stopping[0] = val_loss
                early_stopping[2] = 0
                torch.save({"epoch" : epoch + 1,
                            "net_state_dict" : net.state_dict(),
                            "optimizer_state_dict" : optimizer.state_dict(),
                            "train_loss_list" : train_loss_list,
                            "train_loss_list" : train_loss_list,
                            "val_loss_list" : val_loss_list,
                            }, "save_models/two_stream_trial.pth")
            else:
                early_stopping[2] += 1
                if early_stopping[2] == early_stopping[1]:
                    break

            # tensorboard
            writer.add_scalar("Train Loss", train_loss, epoch + 1)
            writer.add_scalar("Valid Loss", val_loss, epoch + 1)
            print("log updated")

        except ValueError:
            continue

if __name__ == "__main__":
    main()
