import argparse
import glob
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter

import dataset
import fusion
import kptnet
import transformer
import swin_transformer
import swin_transformer_v2

def print_memory_usage():
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(F"GPU Memory Allocated: {allocated:.2f} MB")
    print(F"GPU Memory Reserved: {reserved:.2f} MB")

def train(train_dataloader, swin_t, unet, fuse, loss_function, optimizer, device, bptt, ntokens):
    swin_t.train()
    unet.train()
    fuse.train()
    total_loss =  0
    start_time = time.time()
    # src_mask = transformer.generate_square_subsequent_mask(bptt).to(device)

    num_batches = len(train_dataloader)

    with tqdm(total=len(train_dataloader)) as pbar:
        for data in train_dataloader:
            inputs = data[0]
            kptmap = inputs["kptmap"]
            img = inputs["img"]
            targets = data[1]

            # batch_size = data.size(0)
            # seq_len = data.size(1)
            # src_mask = transformer.generate_square_subsequent_mask(seq_len).to(device)
            # if batch_size != bptt: # only one last batch
                # src_mask = src_mask[:batch_size, :batch_size]

            img = img.to(device)
            kptmap = kptmap.to(device)
            img_pred = swin_t(img)
            kpt_pred = unet(kptmap)
            pred = fuse(img_pred, kpt_pred)
            loss = loss_function(pred, targets.to(device))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(swin_t.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(fuse.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            pbar.update()

    return total_loss / len(train_dataloader)

def evaluate(val_dataloader, swin_t, unet, fuse, loss_function, device, bptt, ntokens):
    swin_t.eval()
    unet.eval()
    fuse.eval()
    total_loss = 0
    # src_mask = transformer.generate_square_subsequent_mask(bptt).to(device)

    with torch.no_grad():
        with tqdm(total=len(val_dataloader)) as pbar:
            for data in val_dataloader:
                inputs = data[0]
                kptmap = inputs["kptmap"]
                img = inputs["img"]
                targets = data[1]
                batch_size = len(data[1])
                # seq_len = data.size(1)
                # src_mask = transformer.generate_square_subsequent_mask(seq_len).to(device)
                # if batch_size != bptt:
                    # src_mask = src_mask[:batch_size, :batch_size]
                img = img.to(device)
                kptmap = kptmap.to(device)
                img_pred = swin_t(img)
                kpt_pred = unet(kptmap)
                pred = fuse(img_pred, kpt_pred)
                # pred_frat = pred.view(-1, ntokens)
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
    lr = 1e-4
    batch_size = args.batch_size

    ntokens = 3840
    emsize = 512
    d_hid = 2048
    nlayers = 6
    nhead = 8
    dropout = 0.1
    bptt = 35

    img_height = 1920
    img_width = 3840


    swin_t = swin_transformer_v2.SwinTransformerV2(img_height=img_height, img_width=img_width,
                                              output_img_size=192*384)
    unet = kptnet.UNet(in_channels=3, out_channels=3)
    fuse = fusion.Fusion(in_channels=6, out_channels=3)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        print("---------- Use GPU ----------")
        # swin_t = nn.DataParallel(swin_t)
        # unet = nn.DataParallel(unet)
        # fuse = nn.DataParallel(fusion)
    else:
        print("---------- Use CPU ----------")
    swin_t.half().to(device)
    unet.half().to(device)
    fuse.half().to(device)

    # loss_function = nn.CrossEntropyLoss()
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(swin_t.parameters(), lr=lr)

    writer = SummaryWriter(log_dir="logs")

    train_loss_list = list()
    val_loss_list = list()

    train_data_dir = "data/train"
    val_data_dir = "data/val"
    H = 1920
    W = 3840
    train_data = dataset.Dataset(train_data_dir, img_height=H, img_width=W, transform=None, is_train=True)
    val_data = dataset.Dataset(val_data_dir, img_height=H, img_width=W, transform=None, is_train=False)

    train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                  num_workers=1)
    val_dataloader = DataLoader(val_data, batch_size=batch_size,
                                num_workers=1)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        swin_t.load_state_dict(checkpoint["model_state_dict"])
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
    for epoch in range(epochs):
        epoch += start_epoch
        print(f"Epoch {epoch+1}\n--------------------")
        try:
            # train
            train_loss = train(train_dataloader, swin_t, unet, fuse,
                               loss_function, optimizer, device, bptt, ntokens)
            train_loss_list.append(train_loss)

            # test
            with torch.no_grad():
                val_loss = evaluate(val_dataloader, swin_t, unet, fuse,
                                    loss_function, device, bptt, ntokens)
                val_loss_list.append(val_loss)

            print("Epoch %d : train_loss %.3f" % (epoch + 1, train_loss))
            print("Epoch %d : val_loss %.3f" % (epoch + 1, val_loss))

            # save model
            torch.save({"epoch" : epoch + 1,
                        "model_state_dict" : model.state_dict(),
                        "optimizer_state_dict" : optimizer.state_dict(),
                        "train_loss_list" : train_loss_list,
                        "val_loss_list" : val_loss_list,
                        }, "save_models/" + str(epoch + 1))

            # tensorboard
            writer.add_scalar("Train Loss", train_loss, epoch + 1)
            writer.add_scalar("Valid Loss", val_loss, epoch + 1)

        except ValueError:
            continue

if __name__ == "__main__":
    main()
