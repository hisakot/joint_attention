import argparse

import cv2
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some integers")
    parser.add_argument("--ue_model", required=False)
    parser.add_argument("--real_model", required=False)
    parser.add_argument("--mix_model", required=False)
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
        ue_train_ssim = list()
        for loss in ue_train_loss_list:
            ue_train.append(loss[0])
            ue_train_ssim.append(1 - loss[3])
        ue_val_loss_list = checkpoint["val_loss_list"]
        ue_val = list()
        ue_val_ssim = list()
        for loss in ue_val_loss_list:
            ue_val.append(loss[0])
            ue_val_ssim.append(1 - loss[3])
        ue_test_loss_list = checkpoint["test_loss_list"]
        ue_test = list()
        ue_test_ssim = list()
        for loss in ue_test_loss_list:
            ue_test.append(loss[0])
            ue_test_ssim.append(1 - loss[3])

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
        real_train_ssim = list()
        for loss in real_train_loss_list:
            real_train.append(loss[0])
            real_train_ssim.append(1 - loss[3])
        real_val_loss_list = checkpoint["val_loss_list"]
        real_val = list()
        real_val_ssim = list()
        for loss in real_val_loss_list:
            real_val.append(loss[0])
            real_val_ssim.append(1 - loss[3])
        real_test_loss_list = checkpoint["test_loss_list"]
        real_test = list()
        real_test_ssim = list()
        for loss in real_test_loss_list:
            real_test.append(loss[0])
            real_test_ssim.append(1 - loss[3])

    if args.mix_model:
        checkpoint = torch.load(args.mix_model)
        '''
        start_epoch = checkpoint["epoch"]
        net.load_state_dict(checkpoint["net_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        '''
        # total loss
        mix_train_loss_list = checkpoint["train_loss_list"]
        mix_train = list()
        mix_train_ssim = list()
        for loss in mix_train_loss_list:
            mix_train.append(loss[0])
            mix_train_ssim.append(1 - loss[3])
        mix_val_loss_list = checkpoint["val_loss_list"]
        mix_val = list()
        mix_val_ssim = list()
        for loss in mix_val_loss_list:
            mix_val.append(loss[0])
            mix_val_ssim.append(1 - loss[3])
        mix_test_loss_list = checkpoint["test_loss_list"]
        mix_test = list()
        mix_test_ssim = list()
        for loss in mix_test_loss_list:
            mix_test.append(loss[0])
            mix_test_ssim.append(1 - loss[3])


    # plt
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(231)

    ue_color = "#e5801a" # orange
    real_color = "#1a80e5" # blue
    # ue_color = "#e61919" # red
    mix_color = "#2de52d" # red

    if args.real_model:
        x = list(range(1, len(real_train)+1, 1))
        ax.plot(x, real_train, label="Real-world Dataset", color=real_color)
    if args.ue_model:
        x = list(range(1, len(ue_train)+1, 1))
        ax.plot(x, ue_train, label="Simulated Dataset", color=ue_color)
    if args.mix_model:
        x = list(range(1, len(mix_train)+1, 1))
        ax.plot(x, mix_train, label="Sim-real Dataset", color=mix_color)
    ax.set_title("Train Loss")
    plt.legend()
    plt.xlim(0, 70)
    plt.ylim(2.0, 5.0)
    plt.xlabel("epochs")
    plt.ylabel("Train Loss")

    bx = fig.add_subplot(232)
    if args.real_model:
        x = list(range(1, len(real_val)+1, 1))
        bx.plot(x, real_val, label="Real-world Dataset", color=real_color)
    if args.ue_model:
        x = list(range(1, len(ue_val)+1, 1))
        bx.plot(x, ue_val, label="Simulated Dataset", color=ue_color)
    if args.mix_model:
        x = list(range(1, len(mix_train)+1, 1))
        bx.plot(x, mix_val, label="Sim-real Dataset", color=mix_color)
    bx.set_title("Validation Loss")
    plt.legend()
    plt.xlim(0, 70)
    plt.ylim(2.0, 5.0)
    plt.xlabel("epochs")
    plt.ylabel("Validation Loss")

    cx = fig.add_subplot(234)
    if args.real_model:
        x = list(range(1, len(real_train_ssim)+1, 1))
        cx.plot(x, real_train_ssim, label="Real-world Dataset", color=real_color)
    if args.ue_model:
        x = list(range(1,len(ue_train_ssim)+1, 1))
        cx.plot(x, ue_train_ssim, label="Simulated Dataset", color=ue_color)
    if args.mix_model:
        x = list(range(1, len(mix_train)+1, 1))
        cx.plot(x, mix_train_ssim, label="Sim-real Dataset", color=mix_color)
    cx.set_title("Train SSIM")
    plt.legend()
    plt.xlim(0, 70)
    plt.ylim(0, 1.0)
    plt.xlabel("epochs")
    plt.ylabel("Train SSIM")

    dx = fig.add_subplot(235)
    if args.real_model:
        x = list(range(1, len(real_val_ssim)+1, 1))
        dx.plot(x, real_val_ssim, label="Real-world Dataset", color=real_color)
    if args.ue_model:
        x = list(range(1,len(ue_val_ssim)+1, 1))
        dx.plot(x, ue_val_ssim, label="Simulated Dataset", color=ue_color)
    if args.mix_model:
        x = list(range(1, len(mix_train)+1, 1))
        dx.plot(x, mix_val_ssim, label="Sim-real Dataset", color=mix_color)
    dx.set_title("Validation SSIM")
    plt.legend()
    plt.xlim(0, 70)
    plt.ylim(0, 1.0)
    plt.xlabel("epochs")
    plt.ylabel("Validation SSIM")

    ex = fig.add_subplot(233)
    if args.real_model:
        x = list(range(1, len(real_test)+1, 1))
        ex.plot(x, real_test, label="Real-world Dataset", color=real_color)
    if args.ue_model:
        x = list(range(1, len(ue_test)+1, 1))
        ex.plot(x, ue_test, label="Simulated Dataset", color=ue_color)
    if args.mix_model:
        x = list(range(1, len(mix_train)+1, 1))
        ex.plot(x, mix_test, label="Sim-real Dataset", color=mix_color)
    ex.set_title("Test Loss")
    plt.legend()
    plt.xlim(0, 70)
    plt.ylim(2.0, 5.0)
    plt.xlabel("epochs")
    plt.ylabel("Loss")

    fx = fig.add_subplot(236)
    if args.real_model:
        x = list(range(1, len(real_test_ssim)+1, 1))
        fx.plot(x, real_test_ssim, label="Real-world Dataset", color=real_color)
    if args.ue_model:
        x = list(range(1, len(ue_test_ssim)+1, 1))
        fx.plot(x, ue_test_ssim, label="Simulated Dataset", color=ue_color)
    if args.mix_model:
        x = list(range(1, len(mix_train)+1, 1))
        fx.plot(x, mix_test_ssim, label="Sim-real Dataset", color=mix_color)
    fx.set_title("Test SSIM")
    plt.legend()
    plt.xlim(0, 70)
    plt.ylim(0, 1.0)
    plt.xlabel("epochs")
    plt.ylabel("SSIM")


    plt.subplots_adjust(hspace=0.3)

    plt.show()
    fig.savefig("./data/loss_graph.png")
    plt.close(fig)
