import numpy as np
import cv2
from matplotlib import pyplot as plt
import res.utils as utils
from scipy.interpolate import spline

from pylab import mpl

# cap = cv2.VideoCapture(1)
#
# if cap.isOpened() is False:
#     print('调用camera失败')
#     exit()
#
#
# while (1):
#     ret, frame = cap.read()
#     cv2.imshow("capture", frame)
#
#     key = cv2.waitKey(3) & 0xFF
#
#     if key == ord('b'):
#         img_bg = cv2.GaussianBlur(frame, (5, 5), 0)
#     if key == ord('p'):
#         img_BGR = cv2.GaussianBlur(frame,(5,5),0)
#         cv2.imshow('im',img_BGR)
#     if key == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

# utils.countYCrCbFromRGB(253,217,208)

#高亮
#142,113
#150,100
#160,99
#136,98
#159,105

#135-162  95-120

#纯白
#252,126,123

#比较暗
#136,114
#144,118
#133,115
#134,111
#146,117
#130-150,108-120

#lower_range = np.array([10, 130, 95])
#upper_range = np.array([250, 170, 120])

#正常光
#lower_range = np.array([30, 125, 95])
#upper_range = np.array([250, 162, 125])

#showimg
# PIC_PATH = '/home/inory/PycharmProjects/Proj_bs/img/'
#
#
# for fileNum in range(6):
#     for picNum in range(10):
#         img = cv2.imread(PIC_PATH + 'img' + str(fileNum) + '/' + str(picNum) +'.jpg',cv2.IMREAD_GRAYSCALE)
#         plt.subplot(6, 10, fileNum*10+picNum+1), plt.imshow(img, cmap='gray'), plt.xticks([]), plt.yticks([])
# plt.show()

#plt accu
# test_accu = np.load('../data/test_accu_5e-4_adam.npy')
# train_accu = np.load('../data/train_accu_5e-4_adam.npy')


cost_1 = np.load('../data/test_accu_5e-4_adam.npy')

x_axis = np.arange(1,121)
plt.xlabel("round")
plt.ylabel("accuracy")


plt.plot(x_axis,cost_1,color='r',linewidth=2,label = 'test accuracy')


plt.xticks(np.linspace(0,130,14))
plt.yticks(np.linspace(0,1,11))
plt.legend()


plt.show()
