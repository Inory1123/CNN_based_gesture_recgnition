# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test1.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        #定义窗口
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 800)
        MainWindow.setMinimumSize(QtCore.QSize(1000, 800))
        MainWindow.setMaximumSize(QtCore.QSize(1000, 800))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        #camFrame
        self.camframe = QtWidgets.QFrame(self.centralwidget)
        self.camframe.setGeometry(QtCore.QRect(40, 30, 600, 500))
        self.camframe.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.camframe.setFrameShadow(QtWidgets.QFrame.Raised)
        self.camframe.setLineWidth(5)
        self.camframe.setMidLineWidth(0)
        self.camframe.setObjectName("frame")

        # camlabel
        self.camlabel = QtWidgets.QLabel(self.camframe)
        self.camlabel.setGeometry(QtCore.QRect(0, 0, 600, 500))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.camlabel.sizePolicy().hasHeightForWidth())
        self.camlabel.setSizePolicy(sizePolicy)
        self.camlabel.setText("")
        self.camlabel.setPixmap(QtGui.QPixmap("test.jpg"))
        self.camlabel.setScaledContents(True)
        self.camlabel.setObjectName("label")

        #rstframe
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(690, 30, 260, 500))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_2.setObjectName("frame_2")

        #rst_img_label
        self.rst_img_label = QtWidgets.QLabel(self.frame_2)
        self.rst_img_label.setGeometry(QtCore.QRect(60, 20, 150, 150))
        self.rst_img_label.setText("")
        self.rst_img_label.setPixmap(QtGui.QPixmap("test.jpg"))
        self.rst_img_label.setScaledContents(True)
        self.rst_img_label.setObjectName("label_2")

        #rst_label1
        self.rst_label1 = QtWidgets.QLabel(self.frame_2)
        self.rst_label1.setEnabled(True)
        self.rst_label1.setGeometry(QtCore.QRect(60, 190, 150, 40))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rst_label1.sizePolicy().hasHeightForWidth())
        self.rst_label1.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setKerning(True)
        self.rst_label1.setFont(font)
        self.rst_label1.setMouseTracking(True)
        self.rst_label1.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.rst_label1.setAcceptDrops(False)
        self.rst_label1.setAutoFillBackground(False)
        self.rst_label1.setStyleSheet("background-color: rgb(170, 170, 127);")
        self.rst_label1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.rst_label1.setTextFormat(QtCore.Qt.AutoText)
        self.rst_label1.setAlignment(QtCore.Qt.AlignCenter)
        self.rst_label1.setObjectName("label_3")


        #rst_label2
        self.rst_label2 = QtWidgets.QLabel(self.frame_2)
        self.rst_label2.setEnabled(True)
        self.rst_label2.setGeometry(QtCore.QRect(60, 240, 150, 40))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rst_label2.sizePolicy().hasHeightForWidth())
        self.rst_label2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setKerning(True)
        self.rst_label2.setFont(font)
        self.rst_label2.setMouseTracking(False)
        self.rst_label2.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.rst_label2.setAcceptDrops(False)
        self.rst_label2.setAutoFillBackground(False)
        self.rst_label2.setStyleSheet("background-color: rgb(170, 170, 127);")
        self.rst_label2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.rst_label2.setAlignment(QtCore.Qt.AlignCenter)
        self.rst_label2.setObjectName("label_4")

        # rst_label3
        self.rst_label3 = QtWidgets.QLabel(self.frame_2)
        self.rst_label3.setEnabled(True)
        self.rst_label3.setGeometry(QtCore.QRect(60, 290, 150, 40))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rst_label3.sizePolicy().hasHeightForWidth())
        self.rst_label3.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setKerning(True)
        self.rst_label3.setFont(font)
        self.rst_label3.setMouseTracking(False)
        self.rst_label3.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.rst_label3.setAcceptDrops(False)
        self.rst_label3.setAutoFillBackground(False)
        self.rst_label3.setStyleSheet("background-color: rgb(170, 170, 127);")
        self.rst_label3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.rst_label3.setAlignment(QtCore.Qt.AlignCenter)
        self.rst_label3.setObjectName("label_5")

        # rst_label4
        self.rst_label4 = QtWidgets.QLabel(self.frame_2)
        self.rst_label4.setEnabled(True)
        self.rst_label4.setGeometry(QtCore.QRect(60, 340, 150, 40))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rst_label4.sizePolicy().hasHeightForWidth())
        self.rst_label4.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setKerning(True)
        self.rst_label4.setFont(font)
        self.rst_label4.setMouseTracking(False)
        self.rst_label4.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.rst_label4.setAcceptDrops(False)
        self.rst_label4.setAutoFillBackground(False)
        self.rst_label4.setStyleSheet("background-color: rgb(170, 170, 127);")
        self.rst_label4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.rst_label4.setAlignment(QtCore.Qt.AlignCenter)
        self.rst_label4.setObjectName("label_6")

        # rst_label5
        self.rst_label5 = QtWidgets.QLabel(self.frame_2)
        self.rst_label5.setEnabled(True)
        self.rst_label5.setGeometry(QtCore.QRect(60, 390, 150, 40))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rst_label5.sizePolicy().hasHeightForWidth())
        self.rst_label5.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setKerning(True)
        self.rst_label5.setFont(font)
        self.rst_label5.setMouseTracking(False)
        self.rst_label5.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.rst_label5.setAcceptDrops(False)
        self.rst_label5.setAutoFillBackground(False)
        self.rst_label5.setStyleSheet("background-color: rgb(170, 170, 127);")
        self.rst_label5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.rst_label5.setAlignment(QtCore.Qt.AlignCenter)
        self.rst_label5.setObjectName("label_7")


        # rst_label6
        self.rst_label6 = QtWidgets.QLabel(self.frame_2)
        self.rst_label6.setEnabled(True)
        self.rst_label6.setGeometry(QtCore.QRect(60, 440, 150, 40))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rst_label6.sizePolicy().hasHeightForWidth())
        self.rst_label6.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setKerning(True)
        self.rst_label6.setFont(font)
        self.rst_label6.setMouseTracking(False)
        self.rst_label6.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.rst_label6.setAcceptDrops(False)
        self.rst_label6.setAutoFillBackground(False)
        self.rst_label6.setStyleSheet("background-color: rgb(170, 170, 127);")
        self.rst_label6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.rst_label6.setAlignment(QtCore.Qt.AlignCenter)
        self.rst_label6.setObjectName("label_8")


        #opframe
        self.opframe = QtWidgets.QFrame(self.centralwidget)
        self.opframe.setGeometry(QtCore.QRect(20, 570, 941, 171))
        self.opframe.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.opframe.setFrameShadow(QtWidgets.QFrame.Raised)
        self.opframe.setObjectName("frame_3")


        self.checkBox = QtWidgets.QCheckBox(self.opframe)
        self.checkBox.setGeometry(QtCore.QRect(40, 40, 130, 40))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.checkBox.setFont(font)
        self.checkBox.setObjectName("checkBox")


        self.checkBox_2 = QtWidgets.QCheckBox(self.opframe)
        self.checkBox_2.setGeometry(QtCore.QRect(40, 90, 130, 40))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.checkBox_2.setFont(font)
        self.checkBox_2.setObjectName("checkBox_2")


        self.button_get_cam = QtWidgets.QPushButton(self.opframe)
        self.button_get_cam.setGeometry(QtCore.QRect(200, 60, 140, 50))

        self.button_get_bg = QtWidgets.QPushButton(self.opframe)
        self.button_get_bg.setGeometry(QtCore.QRect(350, 60, 140, 50))



        self.button_get_gesture = QtWidgets.QPushButton(self.opframe)
        self.button_get_gesture.setGeometry(QtCore.QRect(500, 60, 140, 50))

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "手势识别"))
        self.rst_label1.setText(_translate("MainWindow", "0"))
        self.rst_label2.setText(_translate("MainWindow", "1"))
        self.rst_label3.setText(_translate("MainWindow", "2"))
        self.rst_label4.setText(_translate("MainWindow", "3"))
        self.rst_label5.setText(_translate("MainWindow", "4"))
        self.rst_label6.setText(_translate("MainWindow", "5"))
        self.checkBox.setText(_translate("MainWindow", "采用帧差法"))
        self.checkBox_2.setText(_translate("MainWindow", "自动采集"))
        self.button_get_cam.setText(_translate("MainWindow", "打开camera"))
        self.button_get_bg.setText(_translate("MainWindow", "采集背景图"))
        self.button_get_gesture.setText(_translate("MainWindow", "提取手势"))

