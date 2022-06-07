from ui_main import Ui_MainWindow
import torch
from torch import nn
from skimage import transform
import sys
import os
import numpy as np
import matplotlib

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QSize
from SimpleITK import GetArrayFromImage, ReadImage, WriteImage, GetImageFromArray
from matplotlib import pylab as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
matplotlib.use("Qt5Agg")


def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


class Figure_Canvas(FigureCanvas):
    def __init__(self, width=4, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(Figure_Canvas, self).__init__(self.fig)
        self.ax = self.fig.add_subplot(111)



class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.retranslateUi(self)

        self.niiPath = ""
        self.spacing = []
        self.direction = []
        self.origin = []
        self.width = 0
        self.height = 0
        self.img = np.zeros((1, 1, 1))
        self.target = np.zeros((1, 1, 1))
        self.img_data = np.zeros((1, 1, 1))
        self.imgSize = 256
        self.slice = 6
        self.name = 'C0'
        self.idx1 = 0
        self.idx2 = 0
        self.isOpen = False
        self.isAdjust = False

        self.directs = ["000", "001", "010", "011", "100", "101", "110", "111"]
        self.direct = ""
        self.model = nn.Sequential()

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

        self.model_C0 = net
        self.model_C0.load_state_dict(torch.load(resource_path("./data/Ori_C0.pth"), map_location=torch.device('cpu')))

        self.model_T2 = net
        self.model_T2.load_state_dict(torch.load(resource_path("./data/Ori_T2.pth"), map_location=torch.device('cpu')))

        self.model_LGE = net
        self.model_LGE.load_state_dict(
            torch.load(resource_path("./data/Ori_LGE.pth"), map_location=torch.device('cpu')))

        self.shape = 1

        self.F = Figure_Canvas(width=3, height=2, dpi=100)  # 创建实例

        self.fig = self.F.figure
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")

        self.verticalLayout_6.addWidget(self.F)
        self.openButton.clicked.connect(self.click_open_button)
        self.horizontalSlider.valueChanged.connect(self.move_slider)
        self.adjustButton.clicked.connect(self.click_adjust_button)
        self.saveButton.clicked.connect(self.saveas)

        self.classBox.addItems(['C0', 'T2', 'LGE'])
        self.intruction_label.setPixmap(QPixmap.fromImage(QImage(
            resource_path('./data/instruction.png')).scaled(QSize(270, 500), Qt.IgnoreAspectRatio)))
        self.name_label.setText(self.niiPath)



    def show_image(self, slice_index):
        try:
            itk_img = ReadImage(self.niiPath)
            img_data = GetArrayFromImage(itk_img)
            self.img_data = img_data

            self.width = img_data.shape[1]
            self.height = img_data.shape[2]

            # print(self.img.shape)
            plt.cla()

            self.spacing = itk_img.GetSpacing()
            self.direction = itk_img.GetDirection()
            self.origin = itk_img.GetOrigin()
            # print("img:", self.openPath, "direction:", self.direction)
            self.shape = img_data.shape[0]
            self.idx2 = self.shape
            self.no_all_label.setText(str(self.idx2))
            self.horizontalSlider.setRange(1, img_data.shape[0])
            self.fig = self.F.figure
            self.fig.clear()
            self.ax = self.fig.add_subplot(111)
            self.ax.axis("off")

            self.ax.imshow(img_data[slice_index - 1, :, :], interpolation='nearest', aspect='auto', cmap='gray')
            self.fig.canvas.draw()
        except:
            pass

    def move_slider(self):
        if self.isAdjust is False:
            slice_index = self.horizontalSlider.value()
            self.idx1 = slice_index
            self.no_label.setText(str(self.idx1))
            self.show_image(slice_index)
        else:
            slice_index = self.horizontalSlider.value()
            self.idx1 = slice_index
            self.no_label.setText(str(self.idx1))
            self.rotate(slice_index)

    def click_open_button(self):
        img_name, img_type = QFileDialog.getOpenFileName(self, "Choose your file.", os.getcwd(),
                                                         filter='nii.gz Files (*.nii.gz)')
        self.niiPath = img_name
        if 'C0' in img_name:
            self.name = 'C0'
            self.classBox.setCurrentIndex(0)
        if 'T2' in img_name:
            self.name = 'T2'
            self.classBox.setCurrentIndex(1)
        if 'LGE' in img_name:
            self.name = 'LGE'
            self.classBox.setCurrentIndex(2)
        slice_index = self.horizontalSlider.value()
        self.show_image(slice_index)
        self.name_label.setText(img_name)
        self.direct = self.predict(self.img_data, self.name)
        # print(self.direct)

        self.isOpen = True

    def preprocess(self, img):
        self.slice = img.shape[0]
        new_target_np = np.zeros((self.slice, self.imgSize, self.imgSize))
        for i in range(self.slice):
            data = img[i, :, :]
            data1 = data.reshape((img.shape[1], img.shape[2]))
            new_target_np[i, :, :] = np.array(transform.resize(data1, (self.imgSize, self.imgSize)))
        new_target = torch.tensor(new_target_np, dtype=torch.float32).reshape(
            (self.slice, 1, self.imgSize, self.imgSize))
        return new_target

    def predict(self, img, name):
        if name == "C0":
            self.model = self.model_C0
        if name == "T2":
            self.model = self.model_T2
        if name == "LGE":
            self.model = self.model_LGE

        new_img = self.preprocess(img)
        predictions = self.model(new_img)
        result_dict = dict()
        for i in range(self.slice):
            prediction = predictions[i]
            key = prediction.argmax()
            direct = self.directs[key]
            if direct in result_dict.keys():
                result_dict[direct] += 1
            else:
                result_dict[direct] = 1
        self.direct, count = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)[0]
        return self.direct

    def adjust(self, img):
        target = img
        if self.direct == "000":
            target = img
        if self.direct == "001":
            target = np.flip(img, 2)
        if self.direct == "010":
            target = np.flip(img, 1)
        if self.direct == "011":
            target = np.flip(np.flip(img, 2), 1)
        if self.direct == "100":
            target = img.transpose((0, 2, 1))
        if self.direct == "101":
            target = np.flip(img.transpose((0, 2, 1)), 1)
        if self.direct == "110":
            target = np.flip(img.transpose((0, 2, 1)), 2)
        if self.direct == "111":
            target = np.flip(np.flip(img.transpose((0, 2, 1)), 2), 1)
        return target

    def click_adjust_button(self):
        if self.isOpen:
            self.tableWidget.insertRow(0)
            item = QTableWidgetItem(str(self.width))
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignHCenter)
            self.tableWidget.setItem(0, 0, item)
            item = QTableWidgetItem(str(self.height))
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignHCenter)
            self.tableWidget.setItem(0, 1, item)
            item = QTableWidgetItem(str(self.shape))
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignHCenter)
            self.tableWidget.setItem(0, 2, item)
            item = QTableWidgetItem(str(self.direct))
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignHCenter)
            self.tableWidget.setItem(0, 3, item)

            slice_idx = self.horizontalSlider.value()
            self.rotate(slice_idx)

        else:
            print("Please open the file!")

    def rotate(self, slice_idx):

        self.target = self.adjust(self.img_data)
        plt.cla()
        self.isAdjust = True

        self.idx2 = self.shape
        self.horizontalSlider.setRange(1, self.img_data.shape[0])
        self.fig = self.F.figure
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")

        self.ax.imshow(self.target[slice_idx - 1, :, :], interpolation='nearest', aspect='auto', cmap='gray')
        self.fig.canvas.draw()

    def saveas(self):
        try:
            if self.isAdjust:
                save_path, img_type = QFileDialog.getSaveFileName(self, "", "untitled_" + self.name,
                                                                filter="nii.gz Files (*.nii.gz)")
                save_img = self.target

                img_save = GetImageFromArray(save_img)
                img_save.SetDirection(self.direction)
                img_save.SetOrigin(self.origin)
                img_save.SetSpacing(self.spacing)
                WriteImage(img_save, save_path)

            else:
                pass
        except:
            pass




















































if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())