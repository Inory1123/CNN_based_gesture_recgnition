import numpy as np
import tensorflow as tf
import os
import cv2
import res.utils as utils
import res.CNN_util as cnn_util
from PIL import Image
from PIL import ImageTk
import time
import sys
from PyQt5.QtWidgets import (QWidget, QMessageBox, QHBoxLayout, QFrame, QSplitter, QStyleFactory, QApplication, QLabel, QMainWindow)
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QImage


from res.a import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

LEARNING_RATE = 0.0001
INPUT_NET_PATH = '../NET/round3_5e-4.ckpt'



#create Net
with tf.name_scope('input') as scope:
    xs = tf.placeholder(tf.float32,[None,4096], name='input')  #input 64*64*1
    x_image = tf.reshape(xs, [-1, 64, 64, 1], name='input_image')

with tf.name_scope('Ys') as scope:
    ys = tf.placeholder(tf.float32,[None,6], name='labels')	 #output 6

with tf.name_scope('Layer1') as scope:
#conv1
    with tf.name_scope('Conv1') as scope:
        W_conv1 = cnn_util.weight_variable([5,5,1,16],name='W_conv1')
        b_conv1 = cnn_util.bias_variable([16],name='B_conv1')
        h_conv1 =cnn_util.conv2d(x_image, W_conv1) + b_conv1  # output 64*64*16
    h_pool1=cnn_util.max_pool_2x2(tf.nn.relu(h_conv1,name='Relu1'),name='MaxPool1')						#output 32*32*16

with tf.name_scope('Layer2') as scope:
#conv2
    with tf.name_scope('Conv2') as scope:
        W_conv2=cnn_util.weight_variable([5,5,16,32],name='W_conv2')
        b_conv2=cnn_util.bias_variable([32],name='B_conv2')
        h_conv2=cnn_util.conv2d(h_pool1,W_conv2)+b_conv2 #output 32*32*32
    h_pool2=cnn_util.max_pool_2x2(tf.nn.relu(h_conv2,name='Relu2'),name='MaxPool2')			#output 16*16*32

with tf.name_scope('Layer3') as scope:
#conv3
    with tf.name_scope('Conv3') as scope:
        W_conv3=cnn_util.weight_variable([5,5,32,64],name='W_conv3')
        b_conv3=cnn_util.bias_variable([64],name='B_conv3')
        h_conv3=cnn_util.conv2d(h_pool2,W_conv3)+b_conv3 #output 16*16*64
    h_pool3=cnn_util.max_pool_2x2(tf.nn.relu(h_conv3,name='Relu3'),name='MaxPool3')			#output 8*8*64
    #flat
    h_pool3_flat=tf.reshape(h_pool3,[-1,8*8*64],name='flat')

with tf.name_scope('Fc1_WithDropOut') as scope:
#fc1
    W_fc1=cnn_util.weight_variable([8*8*64,1024],name='W_fc1')
    b_fc1=cnn_util.bias_variable([1024],name='B_fc1')
    keep_prob = tf.placeholder(tf.float32)

    h_fc1=tf.nn.relu(tf.matmul(h_pool3_flat,W_fc1)+b_fc1,name='Relu4')
    h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

with tf.name_scope('Y_pre') as scope:
#output
    W_out=cnn_util.weight_variable([1024,6],name='W_out')
    b_out=cnn_util.bias_variable([6],name='B_out')

    output = tf.matmul(h_fc1_drop,W_out)+b_out

    prediction=tf.nn.softmax((output),name='Softmax')

with tf.name_scope('Cost') as scope:
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(tf.clip_by_value(prediction,1e-8,1)), reduction_indices=[1]))
# cross_entropy = -tf.reduce_sum(ys * tf.log(prediction))
with tf.name_scope('Train') as scope:
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)



class mywindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        #init window
        super(mywindow, self).__init__()
        self.setupUi(self)

        #create session
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, INPUT_NET_PATH)

        #init boolean
        self.isSubMethod = True
        self.isAutoGetPic = True

        #init var
        self.cvimage = None
        self.cvimbg = None
        self.cap = None
        self.rst_labels = (self.rst_label1,self.rst_label2,self.rst_label3,self.rst_label4,self.rst_label5,self.rst_label6)

        #bind timer
        self.bindTimer()
        #bind method
        self.bindMethod()

    def bindTimer(self):
        self.timer_camera = QTimer(self)
        self.timer_camera.timeout.connect(self.show_pic)

        self.timer_rst = QTimer(self)
        self.timer_rst.timeout.connect(self.getGesture)

    def bindMethod(self):
        self.button_get_cam.clicked.connect(self.button1_click)

        self.button_get_bg.clicked.connect(self.button2_click)

        self.checkBox.setCheckState(Qt.Checked)
        self.checkBox.stateChanged.connect(self.state1_change)

        self.checkBox_2.setCheckState(Qt.Checked)
        self.checkBox_2.stateChanged.connect(self.state2_change)

        self.button_get_gesture.clicked.connect(self.button3_click)



    def button1_click(self):
        if self.button_get_cam.text() == '打开camera':
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened() is False:
                QMessageBox.information(self, 'info', '调用camera失败', QMessageBox.Ok)
                return 0
            self.timer_camera.start(10)
            self.button_get_cam.setText('关闭camera')
        elif self.button_get_cam.text() == '关闭camera':
            self.timer_rst.stop()
            self.timer_camera.stop()
            self.cap.release()
            self.button_get_cam.setText('打开camera')
            self.button_get_gesture.setText('提取手势')
            self.camlabel.setPixmap(QtGui.QPixmap("test.jpg"))

    def button2_click(self):
        if self.button_get_cam.text() == '打开camera':
            QMessageBox.information(self,'info','camera未打开',QMessageBox.Ok)
        else:
            self.cvimbg = self.cvimage

    def button3_click(self):
        if self.button_get_cam.text() == '打开camera':
            QMessageBox.information(self, 'info', 'camera未打开', QMessageBox.Ok)
            return 0
        if self.isAutoGetPic is False:
            self.getGesture()
        elif self.button_get_gesture.text() == '提取手势':
            self.timer_rst.start(10)
            self.button_get_gesture.setText('提取中')
        elif self.button_get_gesture.text() == '提取中':
            self.timer_rst.stop()
            self.button_get_gesture.setText('提取手势')


    def state1_change(self):
        self.isSubMethod = not self.isSubMethod

    def state2_change(self):
        self.isAutoGetPic = not self.isAutoGetPic


    def show_pic(self):
        success, self.cvimage = self.cap.read()
        self.cvimage = cv2.GaussianBlur(self.cvimage, (5, 5), 0)
        if not success:
            return 0
        height, width, bytesPerComponent = self.cvimage.shape
        bytesPerLine = bytesPerComponent * width
        frame = cv2.cvtColor(self.cvimage, cv2.COLOR_BGR2RGB)
        image = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.camlabel.setPixmap(QPixmap.fromImage(image))

    def getGesture(self):
        if self.isSubMethod:

            if  self.cvimbg is None:
                QMessageBox.information(self, 'info', '未采集背景图', QMessageBox.Ok)
                self.timer_rst.stop()
                self.button_get_gesture.setText('提取手势')
                return 0

            gray_img = utils.bakSubYCrCb(self.cvimbg, self.cvimage)
            if gray_img is False:
                return 0
        else:
            gray_img = utils.yCrCb(self.cvimage)
            if gray_img is False:
                return 0

        height, width= gray_img.shape
        bytesPerComponent = 1
        bytesPerLine = bytesPerComponent * width
        image = QImage(gray_img.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
        self.rst_img_label.setPixmap(QPixmap.fromImage(image))

        input_x = np.reshape(gray_img, (1, 4096))
        pred = self.sess.run(prediction, feed_dict={xs: input_x, keep_prob: 1})
        rst_num = np.argmax(pred)
        self.updateLabel(rst_num,pred)


    def updateLabel(self,rst_num,pred):
        for i in range(6):
            self.rst_labels[i].setStyleSheet("background-color: rgb(170, 170, 127);")

        self.rst_labels[rst_num].setStyleSheet("background-color: rgb(255, 25, 17);")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = mywindow()
    window.show()
    sys.exit(app.exec_())