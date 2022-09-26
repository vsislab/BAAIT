from __future__ import print_function

from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import os
import imageio


root_dir          = "CamVid/"
data_dir          = os.path.join(root_dir, "camvid/images")  # train data
label_dir         = os.path.join(root_dir, "camvid/labels")  # train label
label_colors_file = os.path.join(root_dir, "label_colors.txt")  # color to label
val_label_file    = os.path.join(root_dir, "val.csv")  # validation file
train_label_file  = os.path.join(root_dir, "train.csv")  # train file


label2color = {}
color2label = {}
label2index = {}
index2label = {}


def divide_train_val(val_rate=0.2, shuffle=True, random_seed=None):
    """从图片文件目录读取图片名并划分训练测试集存入csv文件"""
    data_namelist   = os.listdir(data_dir)
    data_len    = len(data_namelist)
    val_len     = int(data_len * val_rate)

    if random_seed:
        random.seed(random_seed)

    if shuffle:
        data_idx = random.sample(range(data_len), data_len)
    else:
        data_idx = list(range(data_len))
    # 获取训练集和验证集的图片id
    val_idx     = [data_namelist[i] for i in data_idx[:val_len]]
    train_idx   = [data_namelist[i] for i in data_idx[val_len:]]

    # 制作val.csv
    with open(val_label_file, "w+") as f:
        f.write("img,label\n")
        for idx, name in enumerate(val_idx):
            if 'png' not in name:
                continue
            img_name = os.path.join(data_dir, name)
            name = name[:-4] + "_P" + name[-4:]
            lab_name = os.path.join(label_dir, name)
            f.write("{},{}\n".format(img_name, lab_name))
        print("Successfully create val.csv")

    # 制作train.csv
    with open(train_label_file, "w+") as f:
        f.write("img,label\n")
        for idx, name in enumerate(train_idx):
            if 'png' not in name:
                continue
            img_name = os.path.join(data_dir, name)
            name = name[:-4] + "_P" + name[-4:]
            lab_name = os.path.join(label_dir, name)
            f.write("{},{}\n".format(img_name, lab_name))
        print("Successfully create train.csv")

if __name__ == '__main__':
    divide_train_val(random_seed=1)
