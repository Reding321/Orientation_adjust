import torch
import SimpleITK
from torch import nn
import sys
import os
from skimage import transform
import numpy as np

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 64), nn.Sigmoid(),
            nn.Linear(64, 8)
        )

path = r"D:\Project_Orient\data_adjusted\C0\patient30_C0.nii.gz"

itk_img = SimpleITK.ReadImage(path)
img_array = SimpleITK.GetArrayFromImage(itk_img)

net.load_state_dict(torch.load(resource_path("./data/Ori_C0.pth"), map_location=torch.device('cpu')))


b = min(img_array.shape)
new_target_np = np.zeros((b, 256, 256))
for i in range(b):
    data = img_array[i, :, :]
    new_target_np[i, :, :] = np.array(transform.resize(data, (256, 256)))
new_target = torch.tensor(new_target_np, dtype=torch.float32).reshape((b, 1, 256, 256))

print(net(new_target))
