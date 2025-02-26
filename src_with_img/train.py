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
import transformer
import swin_transformer

def train(train_dataloader, model, loss_function, optimizer, device, bptt, ntokens):
    model.train()
    total_loss =  0
    start_time = time.time()
    # src_mask = transformer.generate_square_subsequent_mask(bptt).to(device)

    num_batches = len(train_dataloader)

    with tqdm(total=len(train_dataloader)) as pbar:
        for data, mask, targets, length in train_dataloader:
            batch_size = data.size(0)
            # seq_len = data.size(1)
            # src_mask = transformer.generate_square_subsequent_mask(seq_len).to(device)
            # if batch_size != bptt: # only one last batch
                # src_mask = src_mask[:batch_size, :batch_size]
            
            pred = model(data.to(device))
            loss = loss_function(pred, targets.to(device))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            pbar.update()

    return total_loss / len(train_dataloader)

def evaluate(val_dataloader, model, loss_function, device, bptt, ntokens):
    model.eval()
    total_loss = 0
    # src_mask = transformer.generate_square_subsequent_mask(bptt).to(device)

    with torch.no_grad():
        with tqdm(total=len(val_dataloader)) as pbar:
            for data, mask, targets, length in val_dataloader:
                batch_size = data.size(0)
                # seq_len = data.size(1)
                # src_mask = transformer.generate_square_subsequent_mask(seq_len).to(device)
                # if batch_size != bptt:
                    # src_mask = src_mask[:batch_size, :batch_size]

                pred = model(data.to(divece))
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
    parser.add_argument("--batch_size", required=True, type=int)
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

    model = swin_transformer.SwinTransformer(img_height=1920, img_width=3840,
                                             output_img_size=1920*3840)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        print("---------- Use", torch.cuda.device_count(), "GPUs ----------")
        model = nn.DataParallel(model)
    else:
        print("---------- Use CPU ----------")
    model.to(device)

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
                                  collate_fn=collate_fn, num_workers=4)
    val_dataloader = DataLoader(val_data, batch_size=batch_size,
                                collate_fn=collate_fn, num_workers=4)

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
    for epoch in range(epochs):
        epoch += start_epoch
        print(f"Epoch {epoch+1}\n--------------------")
        try:
            # train
            train_loss = train(train_dataloader, model, loss_function, optimizer, device, bptt, ntokens)
            train_loss_list.append(train_loss)

            # test
            with torch.no_grad():
                val_loss = evaluate(val_dataloader, model, loss_function, device, bptt, ntokens)
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
