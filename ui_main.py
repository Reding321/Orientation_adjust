# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_main.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(987, 868)
        MainWindow.setStyleSheet("background-color: #2ab8d0rgba(85, 255, 255, 155)")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("background-color: rgba(85, 255, 255, 10)")
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.name_label = QtWidgets.QLabel(self.centralwidget)
        self.name_label.setStyleSheet("color: rgb(85, 85, 255);font: 87 9pt \"Arial Black\";")
        self.name_label.setText("")
        self.name_label.setObjectName("name_label")
        self.verticalLayout_4.addWidget(self.name_label)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem1)
        self.openButton = QtWidgets.QPushButton(self.centralwidget)
        self.openButton.setStyleSheet("background-color: rgba(85, 0, 255, 25);font: 10pt \"Adobe ?????? Std R\";")
        self.openButton.setObjectName("openButton")
        self.verticalLayout.addWidget(self.openButton)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem2)
        self.adjustButton = QtWidgets.QPushButton(self.centralwidget)
        self.adjustButton.setStyleSheet("background-color: rgba(85, 0, 255, 25);font: 10pt \"Adobe ?????? Std R\";")
        self.adjustButton.setObjectName("adjustButton")
        self.verticalLayout.addWidget(self.adjustButton)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem3)
        self.saveButton = QtWidgets.QPushButton(self.centralwidget)
        self.saveButton.setStyleSheet("background-color: rgba(85, 0, 255, 25);font: 10pt \"Adobe ?????? Std R\";")
        self.saveButton.setObjectName("saveButton")
        self.verticalLayout.addWidget(self.saveButton)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem4)
        self.verticalLayout.setStretch(0, 3)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 1)
        self.verticalLayout.setStretch(3, 1)
        self.verticalLayout.setStretch(4, 1)
        self.verticalLayout.setStretch(5, 1)
        self.verticalLayout.setStretch(6, 3)
        self.horizontalLayout_3.addLayout(self.verticalLayout)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem5)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.no_slice_label = QtWidgets.QLabel(self.centralwidget)
        self.no_slice_label.setStyleSheet("color: rgb(85, 85, 255);font: 87 11pt \"Arial Black\";")
        self.no_slice_label.setObjectName("no_slice_label")
        self.horizontalLayout.addWidget(self.no_slice_label)
        self.no_label = QtWidgets.QLabel(self.centralwidget)
        self.no_label.setStyleSheet("color: rgb(85, 85, 255);font: 87 11pt \"Arial Black\";")
        self.no_label.setObjectName("no_label")
        self.horizontalLayout.addWidget(self.no_label)
        self.gang_label = QtWidgets.QLabel(self.centralwidget)
        self.gang_label.setStyleSheet("color: rgb(85, 85, 255);font: 87 11pt \"Arial Black\";")
        self.gang_label.setObjectName("gang_label")
        self.horizontalLayout.addWidget(self.gang_label)
        self.no_all_label = QtWidgets.QLabel(self.centralwidget)
        self.no_all_label.setStyleSheet("color: rgb(85, 85, 255);font: 87 11pt \"Arial Black\";")
        self.no_all_label.setObjectName("no_all_label")
        self.horizontalLayout.addWidget(self.no_all_label)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem6)
        self.class_label = QtWidgets.QLabel(self.centralwidget)
        self.class_label.setStyleSheet("color: rgb(85, 85, 255);font: 87 11pt \"Arial Black\";")
        self.class_label.setObjectName("class_label")
        self.horizontalLayout.addWidget(self.class_label)
        self.classBox = QtWidgets.QComboBox(self.centralwidget)
        self.classBox.setStyleSheet("font: 75 9pt \"Arial\";background-color:rgba(85, 85, 255, 150) ;color:rgb(255, 255, 255)")
        self.classBox.setObjectName("classBox")
        self.horizontalLayout.addWidget(self.classBox)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.verticalLayout_2.addLayout(self.verticalLayout_6)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem7)
        self.horizontalSlider = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalLayout_4.addWidget(self.horizontalSlider)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem8)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        spacerItem9 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem9)
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setStyleSheet("background-color: rgba(85, 255, 255, 35)")
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(4)
        self.tableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(3, item)
        self.verticalLayout_2.addWidget(self.tableWidget)
        self.verticalLayout_2.setStretch(0, 3)
        self.verticalLayout_2.setStretch(3, 2)
        self.verticalLayout_3.addLayout(self.verticalLayout_2)
        self.horizontalLayout_3.addLayout(self.verticalLayout_3)
        self.intruction_label = QtWidgets.QLabel(self.centralwidget)
        self.intruction_label.setText("")
        self.intruction_label.setObjectName("intruction_label")
        self.horizontalLayout_3.addWidget(self.intruction_label)
        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(2, 7)
        self.horizontalLayout_3.setStretch(3, 4)
        self.verticalLayout_4.addLayout(self.horizontalLayout_3)
        self.verticalLayout_4.setStretch(0, 1)
        self.verticalLayout_4.setStretch(1, 1)
        self.verticalLayout_4.setStretch(2, 35)
        self.verticalLayout_5.addLayout(self.verticalLayout_4)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.openButton.setText(_translate("MainWindow", "open"))
        self.adjustButton.setText(_translate("MainWindow", "adjust"))
        self.saveButton.setText(_translate("MainWindow", "save"))
        self.no_slice_label.setText(_translate("MainWindow", "No.Slices:"))
        self.no_label.setText(_translate("MainWindow", "0"))
        self.gang_label.setText(_translate("MainWindow", "/"))
        self.no_all_label.setText(_translate("MainWindow", "0"))
        self.class_label.setText(_translate("MainWindow", "Class:"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Width"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Height"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "No.Slices"))
        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "Orientation"))
