import os
import SimpleITK as sitk
import numpy as np
import random
import matplotlib.pyplot as plt
import skimage.transform


class Preprocess:
    def __init__(self):
        self.trainDir = "D:/Project_Orient/data/train_data_LGE/"
        self.validDir = "D:/Project_Orient/data/valid_data_LGE/"
        self.testDir = "D:/Project_Orient/data/test_data_LGE/"
        self.dataDir = "D:/Project_Orient/task_data_classify/LGE/"
        self.dataDir2 = "D:/Project_Orient/test_data/LGE/"
        self.num = 6
        self.imageSize = 256
        self.directs = ["000", "001", "010", "011", "100", "101", "110", "111"]
        self.ratio = {"train": 0.8, "valid": 0.1, "test": 0.1}

    def process_dir(self):
        imgs = []
        for filename in os.listdir(self.dataDir2):
            directs = np.random.choice(self.directs, size=self.num, replace=False)
            for direct in directs:
                target = self.rotate(self.dataDir2 + filename, direct)
                labeled_imgs = self.get_images(target, direct, filename)
                imgs = imgs + labeled_imgs
        return imgs

    def rotate(self, filepath, method):
        itk_img = sitk.ReadImage(filepath)
        img = sitk.GetArrayFromImage(itk_img)
        target = img
        if method == "000":
            target = img   # 000 Target[x,y,z]=Source[x,y,z]
        if method == "001":
            target = np.flip(img, 2)  # 001 Target[x,y,z]=Source[sx-x,y,z]
        if method == "010":
            target = np.flip(img, 1)  # 010 Target[x,y,z]=Source[x,sy-y,z]
        if method == "011":
            target = np.flip(np.flip(img, 2), 1)  # 011 Target[x,y,z]=Source[sx-x,sy-y,z]
        if method == "100":
            target = img.transpose((0, 2, 1))  # 100 Target[x,y,z]=Source[y,x,z]
        if method == "101":
            target = np.flip(img.transpose((0, 2, 1)), 2)  # 101 Target[x,y,z]=Source[sx-y,x,z]
        if method == "110":
            target = np.flip(img.transpose((0, 2, 1)), 1)  # 110 Target[x,y,z]=Source[y,sy-x,z]
        if method == "111":
            target = np.flip(np.flip(img.transpose((0, 2, 1)), 2), 1)  # 111 Target[x,y,z]=Source[sx-y,sy-x,z]
        return target

    def get_images(self, target, method, filename):
        dim = target.shape
        labeled_imgs = []
        for i in range(dim[0]):
            labeled_imgs.append((method, target[i, :, :], str(i) + "_" + filename))
        return labeled_imgs

    def save_data(self):
        imgs = self.process_dir()
        random.shuffle(imgs)
        n = len(imgs)
        for img in imgs[:int(self.ratio["train"] * n)]:
            self.save_img(img, self.trainDir)
        for img in imgs[int(self.ratio["train"] * n):int((self.ratio["train"] + self.ratio["valid"]) * n)]:
            self.save_img(img, self.validDir)
        for img in imgs[int((self.ratio["train"] + self.ratio["valid"]) * n):]:
            self.save_img(img, self.testDir)

    def save_img(self, img, dataDir):
        method, data, filename = img
        img256 = skimage.transform.resize(data, (self.imageSize, self.imageSize))
        f = plt.figure(figsize=(4, 4))
        plt.axis('off')
        plt.imshow(img256, cmap='gray')
        outpath = dataDir + method + "/"
        outfile = filename.replace("nii.gz", "png")
        plt.savefig(outpath + outfile, bbox_inches='tight', pad_inches=0.0)
        plt.close()

    def save_data2(self):
        imgs = self.process_dir()
        random.shuffle(imgs)
        n = len(imgs)
        # for img in imgs[:int(self.ratio["train"] * n)]:
            # self.save_img(img, self.trainDir)
        # for img in imgs[int(self.ratio["train"] * n):int((self.ratio["train"] + self.ratio["valid"]) * n)]:
            # self.save_img(img, self.validDir)
        # for img in imgs[int((self.ratio["train"] + self.ratio["valid"]) * n):]:
            # self.save_img(img, self.testDir)
        for img in imgs[:]:
            self.save_img(img, self.testDir)


Preprocess().save_data2()







