import argparse
import glob
import os
import time
from tqdm import tqdm

import cv2
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
'''
import fusion
import kptnet
import transformer
import swin_transformer
import swin_transformer_v2
import vision_transformer
import resnet
import PJAE_spatiotemporal
import PJAE_spatial
import vis_transformer # only vision transformer BAD
'''
import PJAE_conv
import cnn_transformer
import swin_heatmap

def print_memory_usage():
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(F"GPU Memory Allocated: {allocated:.2f} MB")
    print(F"GPU Memory Reserved: {reserved:.2f} MB")

def train(train_dataloader, model, loss_function, optimizer, device):
    model.train()

    total_loss =  0
    start_time = time.time()

    num_batches = len(train_dataloader)

    with tqdm(total=len(train_dataloader)) as pbar:
        for data in train_dataloader:
            inp = data[0]
            for key, val in inp.items():
                if torch.is_tensor(val):
                    inp[key] = val.to(device)
            img = inp["img"]
            gazecone = inp["gazecone_map"]
            kptmap = inp["kptmap"]
            inputs = torch.cat([img, gazecone, kptmap], dim=1)

            targets = data[1].to(device)

            pred = model(inputs)
            '''
            np_pred = pred.to("cpu").detach().numpy().copy()
            np_pred = np.squeeze(np_pred, 0)
            np_pred = np.transpose(np_pred, (1, 2, 0))
            np_pred *= 255
            np_pred = np_pred.astype(np.uint8)
            cv2.imshow("pred", np_pred)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            '''

            if loss_function[0] == "cos_similarity":
                pred = pred.view(pred.size(0), -1)
                targets = targets.view(targets.size(0), -1)
                cos_loss = F.cosine_similarity(pred, targets)
                loss = (1 - cos_loss).mean()
            elif loss_function[0] == "MSE":
                lossfunc = nn.MSELoss()
                loss = lossfunc(pred, targets)
            elif loss_function[0] == "MAE":
                lossfunc = nn.L1Loss()
                loss = lossfunc(pred, targets)
            elif loss_function[0] == "cos_MSE":
                alpha = loss_function[1]
                mse_loss = F.mse_loss(pred, targets)
                pred = pred.view(pred.size(0), -1)
                targets = targets.view(targets.size(0), -1)
                cos_loss = 1 - F.cosine_similarity(pred, targets).mean()
                loss = alpha * cos_loss + (1 - alpha) * mse_loss
            elif loss_function[0] == "BCE":
                lossfunc = nn.BCEWithLogitsLoss()
                loss = lossfunc(pred, targets)
            else:
                print("Loss function is wrong")

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

    return total_loss / len(train_dataloader)

def evaluate(val_dataloader, model, loss_function, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        with tqdm(total=len(val_dataloader)) as pbar:
            for data in val_dataloader:
                inp = data[0]
                for key, val in inp.items():
                    if torch.is_tensor(val):
                        inp[key] = val.to(device)
                kptmap = inp["kptmap"]
                gazecone = inp["gazecone_map"]
                img = inp["img"]
                inputs = torch.cat([img, gazecone, kptmap], dim=1)

                targets = data[1].to(device)

                pred = model(inputs)

                if loss_function[0] == "cos_similarity":
                    pred = pred.view(pred.size(0), -1)
                    targets = targets.view(targets.size(0), -1)
                    cos_loss = F.cosine_similarity(pred, targets)
                    loss = (1 - cos_loss).mean()
                elif loss_function[0] == "MSE":
                    lossfunc = nn.MSELoss()
                    loss = lossfunc(pred, targets)
                elif loss_function[0] == "MAE":
                    lossfunc = nn.L1Loss()
                    loss = lossfunc(pred, targets)
                elif loss_function[0] == "cos_MSE":
                    alpha = loss_function[1]
                    mse_loss = F.mse_loss(pred, targets)
                    pred = pred.view(pred.size(0), -1)
                    targets = targets.view(targets.size(0), -1)
                    cos_loss = 1 - F.cosine_similarity(pred, targets).mean()
                    loss = alpha * cos_loss + (1 - alpha) * mse_loss
                elif loss_function[0] == "BCE":
                    lossfunc = nn.BCEWithLogitsLoss()
                    loss = lossfunc(pred, targets)
                else:
                    print("Loss function is wrong")

                total_loss += loss.item()
                pbar.update()

    return total_loss / len(val_dataloader)

def collate_fn(batch):
    inputs, labels = zip(*batch)

    length = torch.tensor([x.shape[0] for x in inputs])

    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    attention_mask = (padded_inputs != 0).float()

    labels = torch.stack(labels)
    return padded_inputs, attention_mask, labels, length

def main():

    parser = argparse.ArgumentParser(description="Process some integers")
    parser.add_argument("--batch_size", required=False, default=1, type=int)
    parser.add_argument("--checkpoint", required=False,
                        help="if you want to retry training, write model path")
    args = parser.parse_args()

    cfg = config.Config()

    batch_size = args.batch_size

    lr = cfg.lr

    img_height = cfg.img_height
    img_width = cfg.img_width

    '''
    resnet50 = resnet.ResNet50(pretrained=False, in_ch=4)
    unet = kptnet.UNet(in_channels=3, out_channels=3)
    fuse = fusion.Fusion(in_channels=6, out_channels=3)
    spatiotemporal = PJAE_spatiotemporal.ModelSpatioTemporal(in_ch=2)
    vis_t = vis_transformer.VisionTransformer(in_channels=5, patch_size=4, emb_size=64,
                                              img_H=img_height, img_W=img_width, num_layers=2,
                                              num_heads=2, forward_expansion=4, num_classes=128)
    swin_unet = vision_transformer.SwinUnet(img_height=img_height, img_width=img_width,
                                            in_chans=5, num_classes=1)
    swin_t = swin_transformer_v2.SwinTransformerV2(img_height=img_height, img_width=img_width,
                                                   in_chans=5, output_H=img_height, output_W=img_width)
    '''
    spatial = PJAE_conv.ModelSpatial(in_ch=5)
    cnn_trans = cnn_transformer.CNNTransformer2Heatmap(in_channels=5, 
                                                       img_size=(img_height, img_width),
                                                       output_size=(img_height, img_width))
    # swin_h = swin_heatmap.SimpleSwinHeatmapModel(in_chans=5)
    model = spatial # TODO have to change

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        print("---------- Use GPU ----------")
        model = nn.DataParallel(model)
    else:
        print("---------- Use CPU ----------")
    model.to(device)

    # loss_function = nn.CrossEntropyLoss()
    # loss_function = ["MSE"]
    # loss_function = ["MAE"]
    # loss_function = ["cos_similarity"]
    # loss_function = ["cos_MSE", 0.8]
    loss_function = ["BCE"]
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1**epoch)

    writer = SummaryWriter(log_dir="logs")

    num_cpu = os.cpu_count()
    num_cpu = num_cpu // 8
    print(" number of cpu: ", num_cpu)

    train_loss_list = list()
    val_loss_list = list()

    train_data_dir = "data/train"
    val_data_dir = "data/val"
    train_data = dataset.Dataset(train_data_dir, img_height=img_height, img_width=img_width,
                                 transform=None, is_train=True, inf_rotate=None)
    val_data = dataset.Dataset(val_data_dir, img_height=img_height, img_width=img_width,
                               transform=None, is_train=False, inf_rotate=None)

    train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                  shuffle=True, num_workers=num_cpu, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size,
                                shuffle=False, num_workers=num_cpu, pin_memory=True)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        # spatial.load_state_dict(checkpoint["pjae_spatial_state_dict"])
        model.load_state_dict(checkpoint["cnn_transformer_state_dict"])
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
            train_loss = train(train_dataloader, model, loss_function, optimizer, device)
            train_loss_list.append(train_loss)

            # test
            with torch.no_grad():
                val_loss = evaluate(val_dataloader, model, loss_function, device)
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
                            # "pjae_spatial_state_dict" : spatial.state_dict(),
                            "cnn_transformer_state_dict" : model.state_dict(),
                            "optimizer_state_dict" : optimizer.state_dict(),
                            "train_loss_list" : train_loss_list,
                            "train_loss_list" : train_loss_list,
                            "val_loss_list" : val_loss_list,
                            }, "save_models/newest_model.pth")
            else:
                early_stopping[2] += 1
                if early_stopping[2] == early_stopping[1]:
                    break

            # tensorboard
            writer.add_scalar("Train Loss", train_loss, epoch + 1)
            writer.add_scalar("Valid Loss", val_loss, epoch + 1)
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f"{name}.grad", param.grad, epoch)
                    writer.add_scalar(f"{name}.grad_norm", param.grad.norm(), epoch)
                    writer.add_histogram(f"{name}.weight", param.data, epoch)
            print("log updated")

        except ValueError:
            continue

if __name__ == "__main__":
    main()
