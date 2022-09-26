import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

from models.fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs
from utils.camvid_loader import CamVidDataset
from utils.show_image import idimage2colorimage
import os
import imageio
n_class = 32
root_dir   = "CamVid/"
val_file   = os.path.join(root_dir, "val.csv")
val_data   = CamVidDataset(csv_file=val_file, phase='val', flip_rate=0)
val_loader   = DataLoader(val_data, batch_size=1, num_workers=8, drop_last=True)

model_path = "work_dirs/FCN8s4_epoch600_scheduler-step50-gamma0.5_lr0.0001_momentum0_w_decay1e-05_miou0.11"
model=torch.load(model_path)
print("Successfully loading model from {}".format(model_path))
model = model.cuda()

output_dir = "pred"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists("pred/fcn8s"):
    os.mkdir("pred/fcn8s")

val_csv = "CamVid/val.csv"
val_name = []
with open(val_csv,'r') as f:
    val_name = f.readlines()[1:]  # 去掉第一行表头
val_name = [name[21:34] for name in val_name]
print(val_name)

def val():
    model.eval()
    cnt = 0
    for iter, batch in enumerate(val_loader):
        inputs = batch['X']

        inputs = inputs.cuda()
        t = time.time()
        output = model(inputs)
        
        print(time.time() - t)
        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)
        y = idimage2colorimage(pred[0])
        imageio.imwrite("{}/{}.png".format(output_dir, val_name[cnt]), y)
        print("Finish {}".format(val_name[cnt]))
        cnt += 1


def main():
    val()

if __name__ == "__main__":
    main()
