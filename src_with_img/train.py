import argparse
import glob
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter

import dataset
import transformer
import swin_transformer
import swin_transformer_v2
import kptnet

def train(train_dataloader, model, loss_function, optimizer, device):
    model.train()
    total_loss =  0
    start_time = time.time()

    num_batches = len(train_dataloader)

    with tqdm(total=len(train_dataloader)) as pbar:
        for data, mask, targets, length in train_dataloader:
            batch_size = data.size(0)

            pred = model(data.to(device))
            loss = loss_function(pred, targets.to(device))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            pbar.update()

    return total_loss / len(train_dataloader)

def evaluate(val_dataloader, model, loss_function, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        with tqdm(total=len(val_dataloader)) as pbar:
            for data, mask, targets, length in val_dataloader:
                batch_size = data.size(0)

                pred = model(data.to(device))
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
    parser.add_argument("--batch_size", required=False, type=int, default=1)
    parser.add_argument("--checkpoint", required=False,
                        help="if you want to retry training, write model path")
    args = parser.parse_args()

    batch_size = args.batch_size

    lr = 1e-3
    img_height = 1920
    img_width = 3840

    model = swin_transformer_v2.SwinTransformerV2(img_height=img_height, img_width=img_width,
                                                  embed_dim=96, output_img_size=192*384)
    model = kptnet.UNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        print("---------- Use", torch.cuda.device_count(), "GPUs ----------")
        model = nn.DataParallel(model)
    else:
        print("---------- Use CPU ----------")
    model.half().to(device)

    # loss_function = nn.CrossEntropyLoss()
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    writer = SummaryWriter(log_dir="logs")

    train_loss_list = list()
    val_loss_list = list()

    train_data_dir = "data/train"
    val_data_dir = "data/val"
    train_data = dataset.Dataset(train_data_dir, transform=None, is_train=True)
    val_data = dataset.Dataset(val_data_dir, transform=None, is_train=False)

    train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                  collate_fn=collate_fn, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(val_data, batch_size=batch_size,
                                collate_fn=collate_fn, shuffle=False, num_workers=1)

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
            writer.add_scalar("Validation Loss", tval_loss, i+1)
        print("Reload midel : ", start_epoch, "and restart training")
    else:
        start_epoch = 0

    epochs = 1000
    early_stopping = [np.inf, 10, 0]
    for epoch in range(epochs):
        epoch += start_epoch
        print(f"Epoch {epoch+1}\n--------------------")
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

            # save model
            if val_loss < early_stopping[0]:
                early_stopping[0] = val_loss
                early_stopping[2] = 0
                torch.save({"epoch" : epoch + 1,
                            "model_state_dict" : model.state_dict(),
                            "optimizer_state_dict" : optimizer.state_dict(),
                            "train_loss_list" : train_loss_list,
                            "val_loss_list" : val_loss_list,
                            }, "save_models/only_img_best_unet.pth")
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
