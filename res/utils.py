import numpy as np
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf

def countYCrCbFromRGB(valR : int, valG: int, valB: int):
    delta = 128
    Y = 0.299 * valR + 0.587 * valG + 0.114 * valB
    Cr = (valR - Y) * 0.713 + delta
    Cb = (valB - Y) * 0.564 + delta

    print('(',Y,',',Cr,',',Cb,')')

def getCamPic(picName: str):
    path = '/home/inory/PycharmProjects/Proj_bs/img/' + picName + '.jpg'
    cap = cv2.VideoCapture(1)
    while (1):
        # get a frame
        ret, frame = cap.read()
        # show a frame
        cv2.imshow("capture", frame)
        if cv2.waitKey(3) & 0xFF == ord('q'):
            cv2.imwrite(path, frame)
            break
    cap.release()
    cv2.destroyAllWindows()

def getCamPicWithBlur():
    cap = cv2.VideoCapture(0)
    while (1):
        # get a frame
        ret, frame = cap.read()
        # show a frame
        cv2.imshow("capture", frame)
        if cv2.waitKey(3) & 0xFF == ord('q'):
            dst = cv2.GaussianBlur(frame,(5,5),0)
            break
    cap.release()
    cv2.destroyAllWindows()
    return dst

def testBlur():
    img = cv2.imread('/home/inory/PycharmProjects/Proj_bs/img/im1.jpg', cv2.IMREAD_COLOR)
    # 彩色图像去噪
    dst = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 3, 21)
    # 平均化
    avg = cv2.blur(img, (7, 7))
    # 高斯滤波
    Gaussian = cv2.GaussianBlur(img, (7, 7), 0)
    # 中位数模糊：去除脉冲噪声
    median = cv2.medianBlur(img, 7)
    # 双边滤波：能去除噪声，也可以保留边缘
    bilateral = cv2.bilateralFilter(img, 7, 75, 75)

    # cv2.imshow('图像', img)
    # cv2.imshow('dst', dst)
    # cv2.imshow('avg', avg)
    # cv2.imshow('Gaussian', Gaussian)
    # cv2.imshow('median', median)
    # cv2.imshow('bilateral', bilateral)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    plt.subplot(2, 2, 1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('img'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(cv2.cvtColor(Gaussian, cv2.COLOR_BGR2RGB)), plt.title('GaussianBlur'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB)), plt.title('medianBlur'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB)), plt.title('bilateralFilter'), plt.xticks([]), plt.yticks([])
    plt.show()

def bacSubstract(imgbg,img):

    fgbg = cv2.createBackgroundSubtractorMOG2(2, 50, 0)

    fgbg.apply(imgbg)
    fgmask = fgbg.apply(img)
    return fgmask


def cvImgShowFromPath(title:str,imgPath:str):
    img = cv2.imread(imgPath)
    cv2.imshow(title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cvImgShow(title:str,img):
    cv2.imshow(title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def yCrCb(img_BGR):

    lower_range = np.array([5, 130, 95])
    upper_range = np.array([255, 170, 126])

    img_YCrCb = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    corlor_mask = cv2.inRange(img_YCrCb, lower_range, upper_range)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_mask = cv2.morphologyEx(corlor_mask, cv2.MORPH_CLOSE, kernel)
    open_mask = cv2.morphologyEx(close_mask, cv2.MORPH_CLOSE, kernel)

    (_, cnts, _) = cv2.findContours(open_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return False

    cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img_BGR, (x, y), (x + w, y + h), (0, 255, 0), 5)

    roi_mask = open_mask[y:y + h, x:x + w]
    result = cv2.resize(roi_mask, (64, 64))
    return result
    # plt.subplot(1, 3, 1), plt.imshow(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)), plt.title('img'), plt.xticks(
    #     []), plt.yticks([])
    # plt.subplot(1, 3, 2), plt.imshow(roi_mask, cmap='gray'), plt.title('mask'), plt.xticks([]), plt.yticks([])
    # plt.subplot(1, 3, 3), plt.imshow(resize, cmap='gray'), plt.title('result'), plt.xticks([]), plt.yticks([])
    # plt.show()

def bakSubYCrCb(imgbg,img):
    # 帧差法 + 肤色检测手势提取


    fgmask = bacSubstract(imgbg, img)
    imgfg = cv2.bitwise_and(img, img, mask=fgmask)
    imgfg = cv2.blur(imgfg, (5, 5))

    lower_range = np.array([5, 130, 95])
    upper_range = np.array([250, 170, 126])

    ycrcb = cv2.cvtColor(imgfg, cv2.COLOR_BGR2YCrCb)
    corlor_mask = cv2.inRange(ycrcb, lower_range, upper_range)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_mask = cv2.morphologyEx(corlor_mask, cv2.MORPH_CLOSE, kernel)
    open_mask = cv2.morphologyEx(close_mask, cv2.MORPH_CLOSE, kernel)

    (_, cnts, _) = cv2.findContours(open_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return False

    cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    roi_mask = open_mask[y:y + h, x:x + w]
    result = cv2.resize(roi_mask, (64, 64))
    return result

    # plt.subplot(1, 4, 1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('img'), plt.xticks(
    #     []), plt.yticks([])
    # plt.subplot(1, 4, 2), plt.imshow(cv2.cvtColor(imgfg, cv2.COLOR_BGR2RGB)), plt.title('imgfg'), plt.xticks(
    #     []), plt.yticks([])
    # plt.subplot(1, 4, 3), plt.imshow(open_mask, cmap='gray'), plt.title('mask'), plt.xticks([]), plt.yticks([])
    # plt.subplot(1, 4, 4), plt.imshow(resize, cmap='gray'), plt.title('result'), plt.xticks([]), plt.yticks([])
    # plt.show()