import glob
import json
import os
import time

import cv2
import tkinter as tk
from tkinter import ttk as ttk
import PIL.Image as Image
import PIL.ImageTk as ImageTk

data_dir = "data/test_ann/"
video_name = "ds_020"
mmpose_path = os.path.join(data_dir, "mmpose", video_name + ".json")
img_dir = os.path.join(data_dir, "frames", video_name)

if not os.path.exists(os.path.join(data_dir, "roll_ann")):
    os.mkdir(os.path.join(data_dir, "roll_ann"))
save_json_path = os.path.join(data_dir, "roll_ann", video_name + ".json")

ROLL = ["NotHuman", "Surgeon", "Assistant", "Anesthesiology", "ScrubNurse", "CirculationNurse", "Visitor"]
JA_RATE = ["0.0:own_work", "0.5:ja_sub", "1.0:ja_core"]

with open(mmpose_path) as f:
    data = json.load(f)
    meta_info = data["meta_info"]
    instance_info = data["instance_info"]

class ImageLabelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Annotaion human information")
        self.root.geometry("1920x1080")

        # Image
        self.canvas = tk.Canvas(self.root, width=1000, height=500, bg="gray21")
        self.canvas.place(relx=0.5, rely=0, anchor=tk.N)
        self.label_img = tk.Label(self.root)
        self.label_img.place(relx=0.5, rely=0.48, anchor=tk.N)
        self.label_instance = tk.Label(self.root)
        self.label_instance.place(relx=0.5, rely=0.51, anchor=tk.N)

        # RadioButton
        self.frame_button = tk.Frame(self.root, width=300, height=300)
        self.frame_button.place(relx=0.5, rely=0.55, anchor=tk.N)

        self.frame_roll = tk.LabelFrame(self.root, text="ROLL", width=300, height=300, 
                                        font=("Arial", 15), foreground="RoyalBlue4")
        self.frame_roll.place(relx=0.4, rely=0.55, anchor=tk.NE)

        self.frame_ja_rate = tk.LabelFrame(self.root, text="JA_RATE", width=300, height=300,
                                           font=("Arial", 15), foreground="IndianRed4")
        self.frame_ja_rate.place(relx=0.6, rely=0.55, anchor=tk.NW)

        self.selected_roll = tk.StringVar(value="None")
        self.selected_ja_rate = tk.StringVar(value=0.0)

        for i, roll_text in enumerate(ROLL):
            tk.Radiobutton(self.frame_roll,
                           text=roll_text,
                           variable=self.selected_roll,
                           value=roll_text,
                           font=("Arial", 15),
                           selectcolor="white",
                           fg="RoyalBlue2",
                           activebackground="skyblue",
                           activeforeground="black").place(relx=0.1, y=30+i*30, anchor=tk.NW)
        for i, ja_value in enumerate(JA_RATE):
            tk.Radiobutton(self.frame_ja_rate, 
                           text=ja_value, 
                           variable=self.selected_ja_rate,
                           value=ja_value,
                           font=("Arial", 15),
                           selectcolor="white", 
                           fg="IndianRed1",
                           activebackground="LightPink1",
                           activeforeground="black").place(relx=0.1, y=30+i*30, anchor=tk.NW)

        # Next Button
        self.next_btn = tk.Button(self.frame_button,
                                  text="\nClick to Next\n",
                                  font=("Arial", 15),
                                  command=self.next_instance)
        self.next_btn.place(relx=0.5, rely=0.55, anchor=tk.N)

        '''
        self.roll = tk.Label(self.frame_button,
                             text="Roll: ",
                             font=("Arial", 15))
        self.roll.place(relx=0.6, rely=0.1, anchor=tk.NW)

        self.ja_rate = tk.Label(self.frame_button,
                                text="Rate: ",
                                font=("Arial", 15))
        self.ja_rate.place(relx=0.6, rely=0.2, anchor=tk.NW)
        '''

        self.frame_idx = 0
        self.kpt_idx = 0
        self.added_instances = list()
        self.frame_info = list()

        self.load_instance_data()
        self.show_image()


    def load_instance_data(self):
        self.frame_data = instance_info[self.frame_idx]
        self.frame_id = self.frame_data["frame_id"]
        self.instances = self.frame_data["instances"]
        self.num_people = len(self.instances)
        self.image_path = os.path.join(img_dir, f"{self.frame_id:06}.png")

    def draw_keypoints(self, image, keypoints):
        for x, y in keypoints:
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), thickness=-1)
        return image

    def show_image(self):
        print("--------------------")
        print(f"{self.image_path}: {self.kpt_idx+1}/{self.num_people}")
        img = cv2.imread(self.image_path)
        keypoints = self.instances[self.kpt_idx]["keypoints"]
        img = self.draw_keypoints(img.copy(), keypoints)
        img = cv2.resize(img, (960, 480))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR to RGB
        img_pil = Image.fromarray(img) # RGB to PIL
        self.img_tk = ImageTk.PhotoImage(img_pil) # PIL to ImageTk
        self.canvas.delete("all")
        self.canvas.create_image(500, 250, image=self.img_tk)
        self.label_img.place_forget()
        self.label_img = tk.Label(self.root,
                                  text=self.image_path,
                                  font=("Arial", 12))
        self.label_img.place(relx=0.5, rely=0.48, anchor=tk.N)
        self.label_instance.place_forget()
        text_instance = "{} / {} instances \t Total: {} frames".format(self.kpt_idx+1, self.num_people, len(instance_info))
        self.label_instance = tk.Label(self.root,
                                       text=text_instance,
                                       font=("Arial", 12))
        self.label_instance.place(relx=0.5, rely=0.51, anchor=tk.N)

    def next_instance(self):
        # display selected option
        '''
        self.roll.place_forget()
        self.ja_rate.place_forget()

        self.roll = tk.Label(self.frame_button,
                             text="Roll: "+self.selected_roll.get(),
                             font=("Arial", 15))
        self.roll.place(relx=0.6, rely=0.1, anchor=tk.NW)

        self.ja_rate = tk.Label(self.frame_button,
                                text="Rate: "+self.selected_ja_rate.get(),
                                font=("Arial", 15))
        self.ja_rate.place(relx=0.6, rely=0.2, anchor=tk.NW)
        '''

        # next instance
        if self.selected_roll.get() == "None":
            return

        print(self.selected_roll.get(), self.selected_ja_rate.get())
        keypoints = self.instances[self.kpt_idx]["keypoints"]
        keypoint_scores = self.instances[self.kpt_idx]["keypoint_scores"]
        bbox = self.instances[self.kpt_idx]["bbox"]
        bbox_score = self.instances[self.kpt_idx]["bbox_score"]
        self.added_instances.append({"keypoints" : keypoints,
                                 "keypoint_scores" : keypoint_scores,
                                 "bbox" : bbox,
                                 "bbox_score" : bbox_score,
                                 "roll": self.selected_roll.get(),
                                 "ja_rate": self.selected_ja_rate.get(),
                                 })

        self.kpt_idx += 1
        if self.kpt_idx >= self.num_people:
            self.frame_info.append({"frame_id" : self.frame_id,
                                    "instances" : self.added_instances,
                                    })
            self.added_instances = []
            self.frame_idx += 1
            self.kpt_idx = 0
            if self.frame_idx < len(instance_info):
                self.load_instance_data()
            else:
                self.canvas.delete("all")
                self.canvas.create_text(500, 250, text="Finished\nYou can close this window",
                                        font=("Arial", 16), fill="white")
                save_data = {"meta_info": meta_info,
                             "instance_info" : self.frame_info}
                with open(save_json_path, mode="wt", encoding="utf-8") as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=4)
                self.next_btn.config(state="disabled")
                return

        self.selected_roll.set("None")
        self.show_image()


if __name__ == '__main__':
    start_time = time.time()

    root = tk.Tk()
    app = ImageLabelingApp(root)
    root.mainloop()

    end_time = time.time()
    diff_time = end_time - start_time
    with open(os.path.join(data_dir, "timelog.txt"), 'a') as t:
        t.write(str(diff_time)+"\n")
