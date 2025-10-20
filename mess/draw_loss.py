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
            ue_train.append(loss[0])
        ue_val_loss_list = checkpoint["val_loss_list"]
        ue_val = list()
        for loss in ue_val_loss_list:
            ue_val.append(loss[0])

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
            real_train.append(loss[0])
        real_val_loss_list = checkpoint["val_loss_list"]
        real_val = list()
        for loss in real_val_loss_list:
            real_val.append(loss[0])


    # plt
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    if args.ue_model:
        x = list(range(1, len(ue_train)+1, 1))
        ax.plot(x, ue_train, label="ue_train")
    if args.real_model:
        x = list(range(1, len(real_train)+1, 1))
        ax.plot(x, real_train, label="real_train")

    bx = fig.add_subplot(121)
    if args.ue_model:
        x = list(range(1, len(ue_val)+1, 1))
        print(x)
        print(ue_train)
        ax.plot(x, ue_val, label="ue_val")
    if args.real_model:
        x = list(range(1, len(real_val)+1, 1))
        ax.plot(x, real_val, label="real_val")

    plt.legend()
    plt.show()
