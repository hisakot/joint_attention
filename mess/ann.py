import glob
import json
import os

import cv2
import tkinter as tk

data_dir = "data"
mmpose_items = []

mmpose_paths = glob.glob(data_dir + "/mmpose/*.json")
mmpose_paths.sort()
img_paths = glob.glob(data_dir + "frames/*/*.png")
img_paths.sort()

for file in mmpose_paths:
    with open(json_path) as f:
        data = json.load(f)
        instances = data["instance_info"]
        mmpose.extend(instances)

root = tk.Tk()
root.title("Annotaion human information")
root.geometry("1920x1080")
canvas = tk.Canvas(root, width=960, height=480, bg="white")
canvas.place(x=0, y=0)
root.mainloop()

for idx in range(len(img_paths)):
    img_path = img_paths[idx]
    img = cv2.imread(img_path)
    img = cv2.resize(img, (480, 960))

    mmpose = mmpose_items[idx]
    frame_id = mmpose["frame_id"]
    instances = mmpose["instances"]
    kpts = [] # human_num * keypoints

    for instance in instances:
        keypoints = instance["keypoints"] # 133 human points
        scores = instance["keypoint_scores"]
        if sum(score >= 0.5 for score in scores) > 133 / 5:
            kpts.append(keypoints)

    for kpt in kpts:
        tpl = tuple([int(xy) for xy in kpt])
        one_person = cv2.circle(img, tpl, 2, (0, 255, 0), thickness=-1)
        canvas.create_image(240, 480, image=one_person)
