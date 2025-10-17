import argparse

import cv2
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some integers")
    parser.add_argument("--ue_model", required=False)
    parser.add_argument("--real_model", required=False)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.ue_model:
        checkpoint = torch.load(args.ue_model)
        '''
        start_epoch = checkpoint["epoch"]
        net.load_state_dict(checkpoint["net_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        '''
        # total loss
        ue_train_loss_list = checkpoint["train_loss_list"]
        ue_train = list()
        for loss in ue_train_loss_list:
            ue_train.append(loss)
        ue_val_loss_list = checkpoint["val_loss_list"]
        ue_val = list()
        for loss in ue_val_loss_list:
            ue_val.append(loss)

    if args.real_model:
        checkpoint = torch.load(args.real_model)
        '''
        start_epoch = checkpoint["epoch"]
        net.load_state_dict(checkpoint["net_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        '''
        # total loss
        real_train_loss_list = checkpoint["train_loss_list"]
        real_train = list()
        for loss in real_train_loss_list:
            real_train.append(loss)
        real_val_loss_list = checkpoint["val_loss_list"]
        real_val = list()
        for loss in real_val_loss_list:
            real_val.append(loss)


    # plt
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    if args.ue_model:
        x = list(range(1, len(ue_train)+1, 1))
        ax.plot(x, ue_train, label="ue_train")
    elif args.real_model:
        x = list(range(1, len(real_train)+1, 1))
        ax.plot(x, real_train, label="real_train")

    plt.legend()
    plt.show()
