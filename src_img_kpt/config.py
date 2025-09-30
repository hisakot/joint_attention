
class Config:
    def __init__(self):
        self.img_height = 320
        self.img_width = 640

        self.lr = 1e-4

        self.train_data_dir = "data/ue/train"
        self.val_data_dir = "data/ue/val"
