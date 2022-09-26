from __future__ import print_function

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import random
import os
import imageio

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import utils


root_dir   = "CamVid/"
train_file = os.path.join(root_dir, "train.csv")
val_file   = os.path.join(root_dir, "val.csv")

num_class = 32
means     = np.array([103.939, 116.779, 123.68]) / 255. # BGR三通道的均值
h, w      = 720, 960
train_h   = int(h * 2 / 3)  # 480
train_w   = int(w * 2 / 3)  # 640
val_h     = 704  # 480
val_w     = 960  # 640


class CamVidDataset(Dataset):

    def __init__(self, csv_file, phase, n_class=num_class, crop=True, 
        flip_rate=0.5):
        self.data      = pd.read_csv(csv_file)
        self.means     = means
        self.n_class   = n_class

        self.flip_rate = flip_rate
        self.crop      = crop
        if phase == 'train':
            self.new_h = train_h
            self.new_w = train_w
        elif phase == 'val':
            self.flip_rate = 0.
            self.crop = True
            self.new_h = val_h
            self.new_w = val_w


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name   = self.data.iloc[idx, 0]
        img        = imageio.imread(img_name)
        label_name = self.data.iloc[idx, 1]
        label      = imageio.imread(label_name)

        if self.crop:
            h, w, _ = img.shape
            top   = random.randint(0, h - self.new_h)
            left  = random.randint(0, w - self.new_w)
            img   = img[top:top + self.new_h, left:left + self.new_w]
            label = label[top:top + self.new_h, left:left + self.new_w]

        if random.random() < self.flip_rate:
            img   = np.fliplr(img)
            label = np.fliplr(label)

        # 减去均值
        img = img[:, :, ::-1]  # 切换到BGR
        img = np.transpose(img, (2, 0, 1)) / 255.
        img[0] -= self.means[0]
        img[1] -= self.means[1]
        img[2] -= self.means[2]

        img = torch.from_numpy(img.copy()).float()
        label = torch.from_numpy(label.copy()).long()

        # 转为One-Hot编码
        h, w = label.size()
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][label == c] = 1

        sample = {'X': img, 'Y': target, 'L': label}

        return sample

