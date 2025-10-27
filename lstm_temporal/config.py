
class Config:
    def __init__(self):
        self.img_height = 320
        self.img_width = 640
        self.in_ch = 5

        self.seed = 417
        self.lr = 5e-5
        self.seq_len = 2

        # self.train_data_dir = "data/short_train"
        # self.val_data_dir = "data/short_val"

        self.train_data_dir = "data/ue/train"
        self.val_data_dir = "data/ue/val"

        # self.train_data_dir = "data/mixed/train"
        # self.val_data_dir = "data/mixed/val"

        self.test_data_dir = "data/ue/test"
        # self.test_data_dir = "data/short_test"
        # self.test_data_dir = "data/mixed/test"

