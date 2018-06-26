import numpy as np
import cv2
from matplotlib import pyplot as plt
import res.utils as utils

PIC_NUM = 800
PIC_PATH = '/home/inory/PycharmProjects/Proj_bs/img/img1/'
USE_BACK_SUB = True
index = 400

cap = cv2.VideoCapture(1)

if cap.isOpened() is False:
    print('调用camera失败')
    exit()

while (1):
    ret, frame = cap.read()
    cv2.imshow("capture", frame)

    key = cv2.waitKey(3) & 0xFF

    if key == ord('b'):
        img_bg = cv2.GaussianBlur(frame, (5, 5), 0)
    if key == ord('p'):
        img_BGR = cv2.GaussianBlur(frame,(5,5),0)
        if USE_BACK_SUB:
            gray_img = utils.bakSubYCrCb(img_bg,img_BGR)
        else:
            gray_img = utils.yCrCb(img_BGR)
        cv2.imshow('gray',gray_img)
    if key == ord('o'):
        path = PIC_PATH + str(index) + '.jpg'
        cv2.imwrite(path,gray_img)
        cv2.destroyWindow('gray')
        print('第',index+1,'张图片')
        index = index + 1
        if index == PIC_NUM:
            print('采集完成')
            break
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()









