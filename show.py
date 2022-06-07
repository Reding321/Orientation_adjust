import SimpleITK as sitk
import skimage.io as io
import numpy as np


def read_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data


def show_img(data):
    for i in range(data.shape[0]):
        io.imshow(data[i, :, :], cmap='gray')
        print(i)
        io.show()


path = r"D:\Project_Orient\data_raw\train25_myops_gd\myops_training_102_gd.nii.gz"
path2 = r"D:\Project_Orient\data_raw\train25\myops_training_102_C0.nii.gz"
data = sitk.ReadImage(path2)
data1 = sitk.GetArrayFromImage(data)
show_img(data1)
data2 = sitk.GetArrayFromImage(data)
# print(data1[0, :, :].shape)

