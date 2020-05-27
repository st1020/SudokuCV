#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# OCR 包含tesseract和临近算法(kNN)手写数字识别

import cv2
import numpy as np

def ocr(number_imgs, type=2):
    # type:
    # 1 kNN识别(Mnist 数据集)：准确率较低
    # 2 kNN识别(经过处理的 Mnist 数据集)：准确率较高速度较快(推荐)
    # 3 tesseract-ocr 单字识别：识别速度较慢并可能出现缺字
    # 4 tesseract-ocr 合并识别：识别准确率较高但也会出现缺字
    # 5 自动识别：先进行4，若缺字则进行2和1补全 优先级：4>3>1 若4缺字过多会导致运算量随缺字数呈指数上升，造成内存溢出
    if type == 1:
        global unpack
        from struct import unpack
        return knnOcr(number_imgs, './mnist/train-images-idx3-ubyte', './mnist/train-labels-idx1-ubyte', True)
    elif type == 2:
        return knnOcr(number_imgs, './data/train-images.npy', './data/train-labels.npy', False)
    elif type == 3:
        global pytesseract
        import pytesseract
        return tesseractOcr(number_imgs)
    elif type == 4:
        global pytesseract
        import pytesseract
        return tesseractOcr1(number_imgs)
    elif type == 5:
        global unpack
        global pytesseract
        from struct import unpack
        import pytesseract
        return autoOcr(number_imgs)

# 自动模式
def autoOcr(number_imgs):
    def findHiatus(l, n):
        if n != 0:
            for i in range(len(l)+1):
                t = l.copy()
                t.insert(i, t2[i])
                findHiatus(t, n-1)
        else:
            global allHiatus
            allHiatus.append(l)
    # tesseract-ocr 合并识别
    number_imgs_hstack = np.hstack(number_imgs)
    img_rgb = cv2.cvtColor(cv2.convertScaleAbs(number_imgs_hstack), cv2.COLOR_GRAY2RGB)
    t1 = pytesseract.image_to_string(img_rgb, config='-c tessedit_char_whitelist=123456789', lang='eng')
    if t1.isdecimal() and len(t1) == len(number_imgs):
        return True, list(t1)
    else:
        t1 = [int(i) for i in t1]
        # tesseract-ocr 单字识别
        t2 = []
        for img in number_imgs:
            img_rgb = cv2.cvtColor(cv2.convertScaleAbs(img), cv2.COLOR_GRAY2RGB)
            ocrStr = pytesseract.image_to_string(img_rgb, config='--psm 10 -c tessedit_char_whitelist=123456789', lang='eng')
            if ocrStr.isdecimal():
                t2.append(int(ocrStr))
            else:
                t2.append(0)
        # kNN 识别
        _, k = knnOcr(number_imgs, './mnist/train-images-idx3-ubyte', './mnist/train-labels-idx1-ubyte')
        for i in range(len(t2)):
            if t2[i] == 0:
                t2[i] = k[i]
        # 计算tesseract-ocr合并识别结果t1差的是那几个数
        # 计算所有的可能性种类
        global allHiatus
        allHiatus = []
        findHiatus(t1, len(number_imgs)-len(t1))
        allHiatusSum = list(map(lambda y:sum(map(lambda x: abs(x[0]-x[1]), zip(y, t2))), allHiatus))
        return True, allHiatus[allHiatusSum.index(min(allHiatusSum))]

# tesseract-ocr 单字识别
def tesseractOcr(number_imgs):
    result = []
    for img in number_imgs:
        img_rgb = cv2.cvtColor(cv2.convertScaleAbs(img), cv2.COLOR_GRAY2RGB)
        ocrStr = pytesseract.image_to_string(img_rgb, config='--psm 10 -c tessedit_char_whitelist=123456789', lang='eng')
        if ocrStr.isdecimal():
            result.append(int(ocrStr))
        else:
            return False, None
    return True, result
    
# tesseract-ocr 合并识别
def tesseractOcr1(number_imgs):
    number_imgs_hstack = np.hstack(number_imgs)
    img_rgb = cv2.cvtColor(cv2.convertScaleAbs(number_imgs_hstack), cv2.COLOR_GRAY2RGB)
    ocrStr = pytesseract.image_to_string(img_rgb, config='-c tessedit_char_whitelist=123456789', lang='eng')
    if ocrStr.isdecimal() and len(ocrStr) == len(number_imgs):
        return True, list(ocrStr)
    else:
        return False, None

# 读入图像
def ReadImgFile(filepath):
    with open(filepath, 'rb') as f:
        _, img_num, h, w = unpack('>4I', f.read(16))
        # fromfile()函数读取数据时需要用户指定文件中的元素类型
        img = np.fromfile(f, dtype = np.uint8).reshape(img_num, h * w)
        return img_num, h, w, img

# 读入图像标签
def ReadLableFIle(filepath):
    with open(filepath, 'rb') as f:
        _, img_num = unpack('>2I', f.read(8))
        label = np.fromfile(f, dtype = np.uint8).reshape(img_num, 1)
        return img_num, label

# kNN 识别
def knnOcr(number_imgs, train_img_path, train_label_path, mnist):
    # 读取训练集和测试集的图像和标签
    if mnist:
        train_img_num, train_h, train_w, train_img = ReadImgFile(train_img_path)
        train_label_num, train_label = ReadLableFIle(train_label_path)
    else:
        train_img = np.load(train_img_path)
        train_label = np.load(train_label_path)
        train_h = train_img.shape[1]
        train_w = train_img.shape[2]
        train_img = train_img.reshape(train_img.shape[0], train_h * train_w)
    
    # 将所有数据转成np.float32类型
    train_img = train_img.astype(np.float32)
    train_label = train_label.astype(np.float32)
    
    # 调整输入数据
    # 更改图片大小
    for i in range(len(number_imgs)):
        number_imgs[i] = cv2.resize(number_imgs[i], (train_w, train_h))
    # 更改数组形状
    number_imgs = np.array(number_imgs, dtype=np.float32)
    number_imgs = number_imgs.reshape(-1, train_w*train_h)
    
    # 调用OpenCV的knn实现分类
    knn = cv2.ml.KNearest_create()
    knn.train(train_img, cv2.ml.ROW_SAMPLE, train_label)
    ret, result, neighbours, dist = knn.findNearest(number_imgs, k = 3)
    
    result = [int(i[0]) for i in result]
    return True, result