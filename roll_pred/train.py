import argparse
import glob
import os
import time
from tqdm import tqdm

import cv2
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter

import config
import dataset
import classifier

def print_memory_usage():
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(F"GPU Memory Allocated: {allocated:.2f} MB")
    print(F"GPU Memory Reserved: {reserved:.2f} MB")

def collate_function(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, targets

def train(train_dataloader, model, loss_function, optimizer, device):
    model.train()

    total_loss =  0
    num_people = 0

    with tqdm(total=len(train_dataloader)) as pbar:
        for images, targets in train_dataloader:
            for img, target in zip(images, targets):
                img = img.to(device)
                bboxes = target["bboxes"].to(device)
                labels = target["labels"].to(device)

                pred = model(img.unsqueeze(0), [bboxes])

                if loss_function[0] == "cos_similarity":
                    pred = pred.view(pred.size(0), -1)
                    labels = labels.view(labels.size(0), -1)
                    cos_loss = F.cosine_similarity(pred, labels)
                    loss = (1 - cos_loss).mean()
                elif loss_function[0] == "MSE":
                    lossfunc = nn.MSELoss()
                    loss = lossfunc(pred, labels)
                elif loss_function[0] == "CrossEntropyLoss":
                    labels = torch.argmax(labels, dim=1)
                    lossfunc = nn.CrossEntropyLoss()
                    loss = lossfunc(pred, labels)
                else:
                    print("Loss function is wrong")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_people += 1
            pbar.update(1)

    return total_loss / num_people

def evaluate(val_dataloader, model, loss_function, device):
    model.eval()
    total_loss = 0
    num_people = 0

    with torch.no_grad():
        with tqdm(total=len(val_dataloader)) as pbar:
            for images, targets in val_dataloader:
                for img, target in zip(images, targets):
                    img = img.to(device)
                    bboxes = target["bboxes"].to(device)
                    labels = target["labels"].to(device)

                    pred = model(img.unsqueeze(0), [bboxes])

                    if loss_function[0] == "cos_similarity":
                        pred = pred.view(pred.size(0), -1)
                        labels = labels.view(labels.size(0), -1)
                        cos_loss = F.cosine_similarity(pred, labels)
                        loss = (1 - cos_loss).mean()
                    elif loss_function[0] == "MSE":
                        lossfunc = nn.MSELoss()
                        loss = lossfunc(pred, labels)
                    elif loss_function[0] == "CrossEntropyLoss":
                        labels = torch.argmax(labels, dim=1)
                        lossfunc = nn.CrossEntropyLoss()
                        loss = lossfunc(pred, labels)
                    else:
                        print("Loss function is wrong")

                    total_loss += loss.item()
                    num_people += 1
                pbar.update(1)

    return total_loss / num_people

def main():

    parser = argparse.ArgumentParser(description="Process some integers")
    parser.add_argument("--batch_size", required=False, default=1, type=int)
    parser.add_argument("--checkpoint", required=False,
                        help="if you want to retry training, write model path")
    args = parser.parse_args()
    batch_size = args.batch_size

    cfg = config.Config()
    lr = cfg.lr
    img_height = cfg.img_height
    img_width = cfg.img_width

    model = classifier.ROIClassifier(num_classes=7)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() >= 2:
        print('---------- Use GPUs ----------')
        # model = nn.DataParallel(model)
    else:
        print(f'---------- Use {device} ----------')
    model.to(device)

    loss_function = ["CrossEntropyLoss"]
    # loss_function = ["MSE"]
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)

    writer = SummaryWriter(log_dir="logs")

    num_cpu = os.cpu_count()
    num_cpu = num_cpu // 4
    print(" number of cpu: ", num_cpu)

    train_loss_list = list()
    val_loss_list = list()

    train_data_dir = "data/train"
    val_data_dir = "data/val"
    train_data = dataset.Dataset(train_data_dir, img_height=img_height, img_width=img_width)
    val_data = dataset.Dataset(val_data_dir, img_height=img_height, img_width=img_width)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                  num_workers=num_cpu, pin_memory=True,
                                  collate_fn=collate_function)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                                num_workers=num_cpu, pin_memory=True,
                                collate_fn=collate_function)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        train_loss_list = checkpoint["train_loss_list"]
        val_loss_list = checkpoint["val_loss_list"]
        for i, train_loss in enumerate(train_loss_list):
            writer.add_scalar("Train Loss", train_loss, i+1)
        for i, val_loss in enumerate(val_loss_list):
            writer.add_scalar("Validation Loss", val_loss, i+1)
        print("Reload model : ", start_epoch, "and restart training")
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
                            "model_state_dict" : model.state_dict(),
                            "optimizer_state_dict" : optimizer.state_dict(),
                            "train_loss_list" : train_loss_list,
                            "train_loss_list" : train_loss_list,
                            "val_loss_list" : val_loss_list,
                            }, "save_models/newest_model.pth")
                            # "pjae_spatial_state_dict" : spatial.state_dict(),
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
