import argparse
import glob
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
import PJAE_conv

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
                    loss = nn.MSELoss(pred, targets)
                elif loss_function == "MAE":
                    lossfunc = nn.L1Loss()
                    loss = lossfunc(pred, targets)

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
                        loss = nn.MSELoss(pred, targets)
                    elif loss_function == "MAE":
                        lossfunc = nn.L1Loss()
                        loss = lossfunc(pred, targets)

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

    net = PJAE_conv.ModelSpatial(in_ch=5)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        print("---------- Use GPU ----------")
        net = nn.DataParallel(net)
    else:
        print("---------- Use CPU ----------")
    net.to(device)

    # loss_function = nn.CrossEntropyLoss()
    # loss_function = "MSE"
    # loss_function = "MAE"
    loss_function = "cos_similarity"
    optimizer = optim.SGD(net.parameters(), lr=lr)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1**epoch)

    writer = SummaryWriter(log_dir="logs")

    train_loss_list = list()
    val_loss_list = list()

    train_data_dir = "data/train"
    val_data_dir = "data/val"
    train_data = dataset.Dataset(train_data_dir,
                                 img_height=img_height, img_width=img_width,
                                 seq_len=3, transform=None, is_train=True)
    val_data = dataset.Dataset(val_data_dir,
                               img_height=img_height, img_width=img_width,
                               seq_len=3, transform=None, is_train=False)

    train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                  shuffle=False, num_workers=1) # FIXME shuffle
    val_dataloader = DataLoader(val_data, batch_size=batch_size,
                                shuffle=False, num_workers=1)

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
                            }, "save_models/newest_model.pth")
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
