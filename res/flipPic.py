import numpy as np
import cv2

PIC_PATH = '/home/inory/PycharmProjects/Proj_bs/img/'


for fileNum in range(6):
    for picNum in range(400):
        img = cv2.imread(PIC_PATH + 'img' + str(fileNum) + '/' + str(picNum) +'.jpg',cv2.IMREAD_GRAYSCALE)
        imgflip = cv2.flip(img,1)
        cv2.imwrite(PIC_PATH + 'img' + str(fileNum) + '/' + str(picNum+400) +'.jpg',imgflip)



