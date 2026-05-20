import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np


class ImageDatasetFiveK(Dataset):
    def __init__(self, root_dir, mode):
        """
        root_dir: 数据集根目录
        mode: train or test
        """
        self.root_dir = root_dir
        self.mode = mode

        train_txt_path = os.path.join(root_dir, "train.txt")
        test_txt_path = os.path.join(root_dir, "test.txt")

        with open(train_txt_path, 'r') as f:
            self.train_file_list = [line.strip() for line in f if line.strip()]

        with open(test_txt_path, 'r') as f:
            self.test_file_list = [line.strip() for line in f if line.strip()]

        self.input_dir = os.path.join(root_dir, "input/JPG/480p")
        self.expertC_dir = os.path.join(root_dir, 'expertC/JPG/480p')

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_file_list)
        elif self.mode == 'test':
            return len(self.test_file_list)

    def __getitem__(self, idx):
        if self.mode == "train":
            img_name = self.train_file_list[idx]
            input_path = os.path.join(self.input_dir, img_name + ".jpg")
            img_input = Image.open(input_path).convert('RGB')
            exptC_path = os.path.join(self.expertC_dir, img_name + ".jpg")
            img_exptC = Image.open(exptC_path).convert('RGB')

        elif self.mode == 'test':
            img_name = self.test_file_list[idx]
            input_path = os.path.join(self.input_dir, img_name + ".jpg")
            img_input = Image.open(input_path).convert('RGB')
            exptC_path = os.path.join(self.expertC_dir, img_name + ".jpg")
            img_exptC = Image.open(exptC_path).convert('RGB')

        if self.mode == "train":
            ratio_H = np.random.uniform(0.6, 1.0)
            ratio_W = np.random.uniform(0.6, 1.0)
            W, H = img_input._size
            crop_h = round(H * ratio_H)
            crop_w = round(W * ratio_W)

            i, j, h, w = transforms.RandomCrop.get_params(img_input, output_size=(crop_h, crop_w))
            img_input = TF.crop(img_input, i, j, h, w)
            img_exptC = TF.crop(img_exptC, i, j, h, w)

            if np.random.random() > 0.5:
                img_input = TF.hflip(img_input)
                img_exptC = TF.hflip(img_exptC)

            if np.random.random() > 0.5:
                img_input = TF.vflip(img_input)
                img_exptC = TF.vflip(img_exptC)

            if np.random.random() > 0.5:
                img_input = TF.rotate(img_input, 90)
                img_exptC = TF.rotate(img_exptC, 90)

            # a = np.random.uniform(0.8, 1.2)
            # img_input = TF.adjust_brightness(img_input, a)
            #
            # a = np.random.uniform(0.8, 1.2)
            # img_input = TF.adjust_saturation(img_input, a)

        img_input = TF.to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)

        return {'input': img_input, 'exptC': img_exptC, 'img_name': img_name}