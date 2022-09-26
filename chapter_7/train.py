from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from models.fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs
from models.unet import UNet
from models.DeepLabV3 import DeepLabV3
from utils.camvid_loader import CamVidDataset

from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import os
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES']='0'

n_class    = 32
batch_size = 4
epochs     = 600
lr         = 1e-4
momentum   = 0
w_decay    = 1e-5
step_size  = 50
gamma      = 0.5

"""FCN"""
# vgg_model = VGGNet(requires_grad=True, remove_fc=True)
# model = FCN8s(pretrained_net=vgg_model, n_class=n_class)

"""UNet"""
# model = UNet(n_channels=3, n_classes=n_class)

"""DeepLabV3"""
model = DeepLabV3(
        n_classes=n_class,
        n_blocks=[3, 4, 23, 3],
        atrous_rates=[6, 12, 18],
        multi_grids=[1, 2, 4],
        output_stride=8,
    )

vgg_model = VGGNet(requires_grad=True, remove_fc=True)
model = FCN8s(pretrained_net=vgg_model, n_class=n_class)
configs    = "FCN8s{}_epoch{}_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
print("Configs:", configs)

root_dir   = "CamVid/"
train_file = os.path.join(root_dir, "train.csv")
val_file   = os.path.join(root_dir, "val.csv")

# 储存模型
model_dir = "work_dirs"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, configs)

use_gpu = torch.cuda.is_available()
gpu_id = list(range(torch.cuda.device_count()))

# 加载数据集
train_data = CamVidDataset(
    csv_file=train_file, phase='train')
val_data   = CamVidDataset(
    csv_file=val_file, phase='val', flip_rate=0)
train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, 
    num_workers=8, drop_last=True)
val_loader   = DataLoader(
    val_data, batch_size=1, num_workers=8, drop_last=True)


if use_gpu:
    time_start = time.time()
    # vgg_model = vgg_model.cuda()
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=gpu_id)
    print("Finish cuda loading, time elapsed {:.6f}".format(
        time.time() - time_start))

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=lr, 
    momentum=momentum, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(
    optimizer, step_size=step_size, gamma=gamma)  # 每30epoch消减


def train():
    best_miou = 0
    for epoch in range(epochs):
        time_start = time.time()
        for iter, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            if use_gpu:
                inputs = batch['X'].cuda()
                labels = batch['L'].cuda()

            else:
                inputs, labels = batch['X'], batch['L']


            outputs = model(inputs)
            labels = labels.long()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()
        
        print("Finish epoch {:^3}, time elapsed {:.6f}".format(
            epoch, time.time() - time_start))
        current_miou = val()
        if current_miou > best_miou:
            best_miou = current_miou
            model_save_path = os.path.join(
                model_dir, configs) + "_miou{:.3}".format(best_miou)
            torch.save(model, model_save_path)
            print("model saved")


def val():
    model.eval()
    total_ious = []
    pixel_accs = []
    for iter, batch in enumerate(val_loader):
        inputs = batch['X']
        if use_gpu:
            inputs = inputs.cuda()
            
        output = model(inputs)
        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class)
        pred = pred.argmax(axis=1).reshape(N, h, w)
        target = batch['L'].cpu().numpy().reshape(N, h, w)

        for p, t in zip(pred, target):
            total_ious.append(iou(p, t))
            pixel_accs.append(pixel_acc(p, t))

    # 计算mIoU
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    miou=np.nanmean(ious)
    print("pix_acc: {:.6f} , meanIoU: {:.6f} ".format(pixel_accs, miou))
    return miou


# https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
def iou(pred, target):
    """计算单个类别IoU"""
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # 如果找不到真值，则不参与评估
        else:
            ious.append(float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total


if __name__ == "__main__":
    train()
    val()
