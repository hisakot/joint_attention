import argparse

import cv2
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some integers")
    parser.add_argument("--ue_model", required=False)
    parser.add_argument("--real_model", required=False)
    args = parser.parse_args()

    if args.ue_model:
        checkpoint = torch.load(args.checkpoint)
        '''
        start_epoch = checkpoint["epoch"]
        net.load_state_dict(checkpoint["net_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        '''
        # total loss
        ue_train_loss_list = checkpoint["train_loss_list"][0]
        ue_val_loss_list = checkpoint["val_loss_list"][0]

    if args.real_model:
        checkpoint = torch.load(args.checkpoint)
        '''
        start_epoch = checkpoint["epoch"]
        net.load_state_dict(checkpoint["net_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        '''
        # total loss
        real_train_loss_list = checkpoint["train_loss_list"][0]
        real_val_loss_list = checkpoint["val_loss_list"][0]


    # plt
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    if args.ue_model and args.real_model:
        if len(ue_train_loss_list) > len(real_train_loss_list):
            x = list(range(1, len(ue_train_loss_list)+1, 1))
        else: # ue < real
            x = list(range(1, len(real_train_loss_list)+1, 1))
        ax.plot(x, ue_train_loss_list, label="ue_train")
        ax.plot(x, real_train_loss_list, label="real_train")
    elif args.ue_model:
        x = list(range(1, len(ue_train_loss_list)+1, 1))
        ax.plot(x, ue_train_loss_list, label="ue_train")
    elif args.real_model:
        x = list(range(1, len(real_train_loss_list)+1, 1))
        ax.plot(x, real_train_loss_list, label="real_train")

    plt.legend()
    plt.show()
