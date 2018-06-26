#从图片文件中读取所有图片，保存为 np.array 格式的文件
#输入 images 格式为灰度图 0~200 64*64 每种手势800张图片
#转为0-1 的灰度图 每张图片reshape为一行 （1,4096） 采集4000张图片为训练集，800张图片为测试集

import numpy as np
import cv2

PIC_PATH = '/home/inory/PycharmProjects/Proj_bs/img/'

START = False

for fileNum in range(6):
    for picNum in range(800):
        rsc = (1/255)*cv2.imread(PIC_PATH + 'img' + str(fileNum) + '/' + str(picNum) +'.jpg',cv2.IMREAD_GRAYSCALE)
        img = cv2.flip(rsc,1)
        img = img.reshape((1, 4096))
        img_lable = np.zeros((1,6),dtype=np.float32)
        img_lable[0][fileNum] = 1.0

        line = np.append(img,img_lable,axis=1)
        if START == False:
            START = True
            arrays = line
        else:
            arrays = np.append(arrays,line,axis=0)

np.random.shuffle(arrays)
images_train = arrays[:4000,:4096]
labels_train = arrays[:4000,4096:4102]
images_test = arrays[4000:4800,:4096]
labels_test = arrays[4000:4800,4096:4102]
#
np.save("../new_data/images_train.npy",images_train)
np.save("../new_data/labels_train.npy",labels_train)
np.save("../new_data/images_test.npy",images_test)
np.save("../new_data/labels_test.npy",labels_test)

#训练集中有4000个样本，手势0,1,2,3,4,5的个数为674/660/659/671/675/661
# a = np.zeros(6)
# images_train = np.load('../data/images_train.npy')
# labels_train = np.load('../data/labels_train.npy')
#
# for row in labels_train:
#     num = int(row.dot(np.arange(6).reshape(6,1)))
#     a[num] = a[num] + 1
# print(a)
# print(labels_train[3].dot(np.arange(6).reshape(6,1)))
# cv2.imshow('gray',images_train[3].reshape((64,64)))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
