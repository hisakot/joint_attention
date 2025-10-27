
class Config:
    def __init__(self):
        self.img_height = 320
        self.img_width = 640

        self.lr = 1e-5

        self.train_data_dir = "data/ue/train"
        self.val_data_dir = "data/ue/val"
        # self.test_data_dir = "data/ue/test"
        self.test_data_dir = "data/short_test"
