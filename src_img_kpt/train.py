import argparse
import glob
import time
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter

import config
import dataset
import fusion
import kptnet
import transformer
import swin_transformer
import swin_transformer_v2
import vision_transformer
import resnet
import PJAE_spatiotemporal
import PJAE_spatial

def print_memory_usage():
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(F"GPU Memory Allocated: {allocated:.2f} MB")
    print(F"GPU Memory Reserved: {reserved:.2f} MB")

def train(train_dataloader, spatial, loss_function, optimizer, device):
    '''
    resnet50.train()
    unet.train()
    fuse.train()
    swin_t.train()
    swin_unet.train()
    spatiotemporal.train()
    '''
    spatial.train()
    total_loss =  0
    start_time = time.time()

    num_batches = len(train_dataloader)

    with tqdm(total=len(train_dataloader)) as pbar:
        for data in train_dataloader:
            inputs = data[0]
            kptmap = inputs["kptmap"]
            gazelinemap = inputs["gazeline_map"]
            gazeconemap = inputs["gazecone_map"]
            saliencymap = inputs["saliency_map"]
            img = inputs["img"]
            targets = data[1]

            '''
            img = img.to(device)
            gazelinemap = gazelinemap.to(device)
            kptmap = kptmap.to(device)
            img_pred = swin_t(img)
            kpt_pred = unet(kptmap)
            pred = fuse(img_pred, kpt_pred)
            '''
            concat_list = [img, gazeconemap]
            concat = torch.cat(concat_list, dim=1)
            concat = concat.to(device)
            pred = spatial(concat)
            loss = loss_function(pred, targets.to(device))

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

def evaluate(val_dataloader, spatial, loss_function, device):
    '''
    resnet50.eval()
    unet.eval()
    fuse.eval()
    swin_t.eval()
    swin_unet.eval()
    spatiotemporal.eval()
    '''
    spatial.eval()
    total_loss = 0

    with torch.no_grad():
        with tqdm(total=len(val_dataloader)) as pbar:
            for data in val_dataloader:
                inputs = data[0]
                kptmap = inputs["kptmap"]
                gazelinemap = inputs["gazeline_map"]
                gazeconemap = inputs["gazecone_map"]
                saliencymap = inputs["saliency_map"]
                img = inputs["img"]
                targets = data[1]
                batch_size = len(data[1])

                '''
                img = img.to(device)
                kptmap = kptmap.to(device)
                '''
                concat_list = [img, gazeconemap]
                concat = torch.cat(concat_list, dim=1)
                concat = concat.to(device)
                pred = spatial(concat)

                '''
                img_pred = swin_t(img)
                kpt_pred = swin_t(kptmap)
                pred = fuse(img_pred, kpt_pred)
                '''
                total_loss += batch_size * loss_function(pred, targets.to(device)).item()
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

    resnet50 = resnet.ResNet50(pretrained=False, in_ch=4)
    swin_t = swin_transformer_v2.SwinTransformerV2(img_height=img_height, img_width=img_width,
                                                   in_chans=4, output_H=img_height, output_W=img_width)
    swin_unet = vision_transformer.SwinUnet(img_height=img_height, img_width=img_width,
                                            in_chans=4, num_classes=3)
    unet = kptnet.UNet(in_channels=3, out_channels=3)
    fuse = fusion.Fusion(in_channels=6, out_channels=3)
    spatiotemporal = PJAE_spatiotemporal.ModelSpatioTemporal(in_ch=4)
    spatial = PJAE_spatial.ModelSpatial(in_ch=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        print("---------- Use GPU ----------")
        # swin_t = nn.DataParallel(swin_t)
        # unet = nn.DataParallel(unet)
        # fuse = nn.DataParallel(fusion)
    else:
        print("---------- Use CPU ----------")
    resnet50.half().to(device)
    swin_t.half().to(device)
    swin_unet.half().to(device)
    unet.half().to(device)
    fuse.half().to(device)
    spatiotemporal.half().to(device)
    spatial.half().to(device)

    # loss_function = nn.CrossEntropyLoss()
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(spatial.parameters(), lr=lr)

    writer = SummaryWriter(log_dir="logs")

    train_loss_list = list()
    val_loss_list = list()

    train_data_dir = "data/train"
    val_data_dir = "data/val"
    train_data = dataset.Dataset(train_data_dir, img_height=img_height, 
                                 img_width=img_width, transform=None, is_train=True)
    val_data = dataset.Dataset(val_data_dir, img_height=img_height, 
                               img_width=img_width, transform=None, is_train=False)

    train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                  shuffle=True, num_workers=1)
    val_dataloader = DataLoader(val_data, batch_size=batch_size,
                                shuffle=False, num_workers=1)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        resnet50.load_state_dict(checkpoint["resnet_state_dict"])
        swin_t.load_state_dict(checkpoint["swin_t_state_dict"])
        swin_unet.load_state_dict(checkpoint["swin_unet_state_dict"])
        unet.load_state_dict(checkpoint["unet_state_dict"])
        fuse.load_state_dict(checkpoint["fuse_state_dict"])
        spatiotemporal.load_state_dict(checkpoint["pjae_spatiotemporal_state_dict"])
        spatial.load_state_dict(checkpoint["pjae_spatial_state_dict"])
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
    early_stopping = [np.inf, 10, 0]
    for epoch in range(epochs):
        epoch += start_epoch
        print(f"--------------------\nEpoch {epoch+1}")
        print(early_stopping)
        try:
            # train
            train_loss = train(train_dataloader, spatial,
                               loss_function, optimizer, device)
            train_loss_list.append(train_loss)

            # test
            with torch.no_grad():
                val_loss = evaluate(val_dataloader, spatial,
                                    loss_function, device)
                val_loss_list.append(val_loss)

            print("Epoch %d : train_loss %.3f" % (epoch + 1, train_loss))
            print("Epoch %d : val_loss %.3f" % (epoch + 1, val_loss))

            # save model
            if val_loss < early_stopping[0]:
                early_stopping[0] = val_loss
                early_stopping[2] = 0
                torch.save({"epoch" : epoch + 1,
                            "resnet_state_dict" : resnet50.state_dict(),
                            "swin_t_state_dict" : swin_t.state_dict(),
                            "swin_unet_state_dict" : swin_unet.state_dict(),
                            "unet_state_dict" : unet.state_dict(),
                            "fuse_state_dict" : fuse.state_dict(),
                            "pjae_spatiotemporal_state_dict" : spatiotemporal.state_dict(),
                            "pjae_spatial_state_dict" : spatial.state_dict(),
                            "optimizer_state_dict" : optimizer.state_dict(),
                            "train_loss_list" : train_loss_list,
                            "train_loss_list" : train_loss_list,
                            "val_loss_list" : val_loss_list,
                            }, "save_models/img_gazecone_pjae_best.pth")
            else:
                early_stopping[2] += 1
                if early_stopping[2] == early_stopping[1]:
                    break

            # tensorboard
            writer.add_scalar("Train Loss", train_loss, epoch + 1)
            writer.add_scalar("Valid Loss", val_loss, epoch + 1)

        except ValueError:
            continue

if __name__ == "__main__":
    main()
