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


def copy_geometry(image: sitk.Image, ref: sitk.Image):
    image.SetOrigin(ref.GetOrigin())
    image.SetDirection(ref.GetDirection())
    image.SetSpacing(ref.GetSpacing())
    return image


img = sitk.ReadImage(r"C:\Users\HP\Desktop\pythonProject\CMRadjustNet\data_new\T2\patient45_T2.nii.gz")
data = sitk.GetArrayFromImage(img)
origin = img.GetOrigin()
direction = img.GetDirection()
space = img.GetSpacing()

data1 = np.flip(data, 1)
savedImg = sitk.GetImageFromArray(data1)
savedImg.SetOrigin(origin)
savedImg.SetDirection(direction)
savedImg.SetSpacing(space)

# sitk.WriteImage(savedImg, "patient45_T2.nii.gz")




