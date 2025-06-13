import glob
import json
import os

import cv2
import tkinter as tk
from tkinter import ttk as ttk
import PIL.Image as Image
import PIL.ImageTk as ImageTk

data_dir = "data/train/"
video_name = "020"
mmpose_path = data_dir + "mmpose/ds_" + video_name + ".json"
img_dir = data_dir + "frames/frames" + video_name

save_jaon_path = "data/train/roll_ann/ds" + video_name + ".json"

ROLL = ["Surgeon", "Assistant", "Anesthesiology", "ScrubNurse", "CirculationNurse", "Visitor", "NotHuman"]
JA_RATE = ["0.0:own_work", "0.5:ja_sub", "1.0:ja_core"]

with open(mmpose_path) as f:
    data = json.load(f)
    instance_info = data["instance_info"]

class ImageLabelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Annotaion human information")
        self.root.geometry("1920x1080")

        # Image
        self.canvas = tk.Canvas(self.root, width=1000, height=500, bg="white")
        self.canvas.place(x=0, y=0)
        self.label_img = tk.Label(self.root)
        self.label_img.place(relx=0.0, rely=0.5)
        self.label_instance = tk.Label(self.root)
        self.label_instance.place(relx=0.3, rely=0.5)

        # RadioButton
        self.frame_roll = tk.LabelFrame(self.root, text="ROLL", width=300, height=300, 
                                        font=("Arial", 15), foreground="skyblue")
        self.frame_roll.place(relx=0.1, rely=0.55)

        self.frame_ja_rate = tk.LabelFrame(self.root, text="JA_RATE", width=300, height=300,
                                           font=("Arial", 15), foreground="skyblue")
        self.frame_ja_rate.place(relx=0.3, rely=0.55)

        self.frame_button = tk.Frame(self.root, width=600, height=300)
        self.frame_button.place(relx=0.1, rely=0.85)

        self.selected_roll = tk.StringVar(value="None")
        self.selected_ja_rate = tk.StringVar(value=0.0)

        for i, roll_text in enumerate(ROLL):
            tk.Radiobutton(self.frame_roll,
                           text=roll_text,
                           variable=self.selected_roll,
                           value=roll_text,
                           font=("Arial", 15),
                           selectcolor="black",
                           fg="white",
                           activebackground="white",
                           activeforeground="black").place(relx=0.1, y=30+i*30, anchor=tk.NW)
        for i, ja_value in enumerate(JA_RATE):
            tk.Radiobutton(self.frame_ja_rate, 
                           text=ja_value, 
                           variable=self.selected_ja_rate,
                           value=ja_value,
                           font=("Arial", 15),
                           selectcolor="black", 
                           fg="white",
                           activebackground="white",
                           activeforeground="black").place(relx=0.1, y=30+i*30, anchor=tk.NW)

        # Next Button
        self.next_btn = tk.Button(self.frame_button,
                                  text="Display selected options",
                                  font=("Arial", 15),
                                  command=self.next_instance)
        self.next_btn.place(relx=0.3, rely=0.05, anchor=tk.NW)

        self.roll = tk.Label(self.frame_button,
                             text="Roll: ",
                             font=("Arial", 15))
        self.roll.place(relx=0.3, rely=0.2, anchor=tk.NW)

        self.ja_rate = tk.Label(self.frame_button,
                                text="Joint Attention contribution: ",
                                font=("Arial", 15))
        self.ja_rate.place(relx=0.3, rely=0.3, anchor=tk.NW)

        self.frame_idx = 0
        self.kpt_idx = 0
        self.annotations = []

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
        print(self.image_path)
        print(self.kpt_idx, len(instance_info), self.frame_idx)
        print(self.annotations)
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
                                  font=("Arial", 10))
        self.label_img.place(relx=0.1, rely=0.5)
        self.label_instance.place_forget()
        text_instance = "{} / {} instances \t Total: {} frames".format(self.kpt_idx, self.num_people, len(instance_info))
        self.label_instance = tk.Label(self.root,
                                       text=text_instance,
                                       font=("Arial", 10))
        self.label_instance.place(relx=0.3, rely=0.5)

    def next_instance(self):
        # display selected option
        self.roll.place_forget()
        self.ja_rate.place_forget()

        self.roll = tk.Label(self.frame_button,
                             text="Roll: "+self.selected_roll.get(),
                             font=("Arial", 15))
        self.roll.place(relx=0.3, rely=0.2, anchor=tk.NW)

        self.ja_rate = tk.Label(self.frame_button,
                                text="Joint Attention contribution: "+self.selected_ja_rate.get(),
                                font=("Arial", 15))
        self.ja_rate.place(relx=0.3, rely=0.3, anchor=tk.NW)

        # next instance
        if self.selected_roll.get() == "None":
            return

        self.annotations.append({
            "frame_id" : self.frame_id,
            "instance_idx" : self.kpt_idx,
            "roll": self.selected_roll.get(),
            "ja_rate": self.selected_ja_rate.get(),
            })

        self.kpt_idx += 1
        if self.kpt_idx >= self.num_people:
            self.frame_idx += 1
            self.kpt_idx = 0
            if self.frame_idx < len(instance_info):
                self.load_instance_data()
            else:
                self.canvas.delete("all")
                self.canvas.create_text(500, 250, text="Finished", font=("Arial", 16))
                print(self.annotations)
                save_data = {"ann_info": self.annotations}
                with open(save_jaon_path, mode="wt", encoding="utf-8") as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=4)
                self.next_btn.config(state="disabled")
                return

        self.selected_roll.set("None")
        self.show_image()


if __name__ == '__main__':
    root = tk.Tk()
    app = ImageLabelingApp(root)
    root.mainloop()
