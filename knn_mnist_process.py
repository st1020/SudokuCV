#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 临近算法(kNN) Mnist 数据集处理

from struct import unpack

import cv2
import numpy as np

def extractNumber(img_number):
    # 找到外接矩形面积最大的轮廓
    contours, _ = cv2.findContours(img_number, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 1:
        x, y, w, h = cv2.boundingRect(contours[0])
    else:
        rects = []
        for cnt in contours:
            rects.append(cv2.boundingRect(cnt))
        x, y, w, h = sorted(rects, key=lambda i:i[2]*i[3])[-1]
    #x, y, w, h = x-1, y-1, w+2, h+2
    img_number_roi = img_number[y:y+h, x:x+w]
    if h > w:
        img_return = np.zeros(shape=(h+2, h+2))
        img_return[1:h+1, (h-w)//2+1:(h-w)//2+w+1] = img_number_roi
    else:
        img_return = np.zeros(shape=(w+2, w+2))
        img_return[(w-h)//2+1:(w-h)//2+h+1, 1:w+1] = img_number_roi
    return cv2.resize(img_return, (50, 50))

# 读入图像
def ReadImgFile(filepath):
    with open(filepath, 'rb') as f:
        _, img_num, h, w = unpack('>4I', f.read(16))
        # fromfile()函数读取数据时需要用户指定文件中的元素类型
        img = np.fromfile(f, dtype = np.uint8).reshape(img_num, h, w)
        return img_num, h, w, img

# 读入图像标签
def ReadLableFIle(filepath):
    with open(filepath, 'rb') as f:
        _, img_num = unpack('>2I', f.read(8))
        label = np.fromfile(f, dtype = np.uint8).reshape(img_num, 1)
        return img_num, label

# 读取训练集和测试集的图像和标签
train_img_num, train_h, train_w, train_img = ReadImgFile('./mnist/train-images-idx3-ubyte')
train_label_num, train_label = ReadLableFIle('./mnist/train-labels-idx1-ubyte')
test_img_num, test_h, test_w, test_img = ReadImgFile('./mnist/t10k-images-idx3-ubyte')
test_label_num, test_label = ReadLableFIle('./mnist/t10k-labels-idx1-ubyte')

imgs_np = []
labels_np = []
for i in range(train_img_num):
    if train_label[i][0] != 0:
        imgs_np.append(extractNumber(train_img[i]))
        labels_np.append([train_label[i][0]])
np.save('./data/train-images', np.asarray(imgs_np, dtype = np.uint8))
np.save('./data/train-labels', np.asarray(labels_np, dtype = np.uint8))

imgs_np = []
labels_np = []
for i in range(test_img_num):
    if test_label[i][0] != 0:
        imgs_np.append(extractNumber(test_img[i]))
        labels_np.append([test_label[i][0]])
np.save('./data/test-images', np.asarray(imgs_np, dtype = np.uint8))
np.save('./data/test-labels', np.asarray(labels_np, dtype = np.uint8))
