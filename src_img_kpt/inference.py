import argparse
import glob
import json
import os
import time

import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter

import dataset
import transformer


def test(test_data, model, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        print(test_data.size())
        test_data = torch.reshape(test_data, (1, test_data.size(0)))
        seq_len = test_data.size(1)
        src_mask = transformer.generate_square_subsequent_mask(seq_len).to(device)
        pred = model(test_data, src_mask)

    return pred

def main():

    parser = argparse.ArgumentParser(description="Process some integers")
    parser.add_argument("--model", required=True, help="Write model path")
    args = parser.parse_args()

    ntokens = 3840
    emsize = 512
    d_hid = 2048
    nlayers = 6
    nhead = 8
    dropout = 0.1
    bptt = 35

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        print("---------- Use GPU ----------")
    else:
        print("---------- Use CPU ----------")

    model = transformer.TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout)

    checkpoint = torch.load(args.model)
    if torch.cuda.device_count() >= 1:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        from collections import OrderedDict
        state_dict = OrderedDict()
        for k, v in checkpoint["model_state_dict"].items():
            name = k[7:] # remove "module."
            state_dict[name] = v
        model.load_state_dict(state_dict)
    model.eval()

    test_data_dir = "data/test"
    json_paths = glob.glob(test_data_dir + "/*.json")
    datas = []
    for json_path in json_paths:
        with open(json_path) as f:
            data = json.load(f)
            instance_info = data["instance_info"]
            datas.extend(instance_info)

    img_paths = glob.glob("data/test/*.png")

    for i, data in enumerate(datas):
        instances = data["instances"]
        inputs = []

        for instance in instances:
            keypoints = instance["keypoints"]
            face_kpt = keypoints[23:91]
            scores = instance["keypoint_scores"]
            face_scores = scores[23:91]

            new_face_kpt = []
            if sum(score >= 0.5 for score in face_scores) > 68 / 5:
                p1, p2, yaw, pitch, roll = dataset.get_head_direction(face_kpt)
                pt1_x, pt1_y, pt2_x, pt2_y = dataset.get_face_rectangle(face_kpt, yaw, pitch)

                pt1_x = max(int(pt1_x), 0)
                pt1_y = max(int(pt1_y), 0)
                pt2_x = min(int(pt2_x), 3839)
                pt2_y = min(int(pt2_y), 1919)
                gz1_x = int(max(0, min(p1[0], 3839)))
                gz1_y = int(max(0, min(p1[1], 1919)))
                gz2_x = int(max(0, min(p2[0], 3839)))
                gz2_y = int(max(0, min(p2[1], 1919)))
                face_rec_gaze = [pt1_x, pt1_y, pt2_x, pt2_y, gz1_x, gz1_y, gz2_x, gz2_y]
                inputs.extend(face_rec_gaze)
        
        test_data = torch.tensor(inputs, dtype=torch.long)

        pred = test(test_data, model, device)
        print(pred)

        img = cv2.imread(img_paths[i])
        x = int(pred[0][0] * 3840)
        y = int(pred[0][1] * 1920)
        cv2.circle(img, (x, y), 30, (0, 0, 255), 2)
        cv2.imwrite("data/test/gaze_" + os.path.basename(img_paths[i]), img)

if __name__ == "__main__":
    main()
