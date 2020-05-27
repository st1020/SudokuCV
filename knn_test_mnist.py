#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 临近算法(kNN)手写数字识别

from struct import unpack

import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# 读取训练集和测试集的图像和标签
train_img_num, train_h, train_w, train_img = ReadImgFile('./mnist/train-images-idx3-ubyte')
train_label_num, train_label = ReadLableFIle('./mnist/train-labels-idx1-ubyte')
test_img_num, test_h, test_w, test_img = ReadImgFile('./mnist/t10k-images-idx3-ubyte')
test_label_num, test_label = ReadLableFIle('./mnist/t10k-labels-idx1-ubyte')

# 显示图像
def Display(img, label, h, w, num, name='test'):
    fig = plt.figure()    # 使用figure()命令来产生一个图
    for i in range(num):
        im = img[i].reshape([h, w])    # 将一维的像素矩阵reshape成原图像大小的二维矩阵
        # add_subplot把图分割成多个子图，三个参数分别为行数、列数、当前子图位置
        ax = fig.add_subplot(1, num, i + 1)
        ax.set_title(str(label[i]))    # 每个子图的命名为其标签
        ax.axis('off')    # 隐藏坐标
        ax.imshow(im, cmap='gray')
    plt.savefig(name+'.png')

#Display(train_img, train_label, train_h, train_w, 5)    # 显示训练集前5张图像
#Display(test_img, test_label, test_h, test_w, 5)    # 显示测试集前5张图像

# 将所有数据转成np.float32类型
train_img = train_img.astype(np.float32)
train_label = train_label.astype(np.float32)
test_img = test_img.astype(np.float32)[5000:10000]
test_label = test_label.astype(np.float32)[5000:10000]

# 调用OpenCV的knn实现分类
knn = cv2.ml.KNearest_create()
knn.train(train_img, cv2.ml.ROW_SAMPLE, train_label)

# 测试1到20所有K值并计算预测准确率
for i in range(1,21):
    ret, result, neighbours, dist = knn.findNearest(test_img, k = i)
    
    # 计算预测准确率
    matches = result == test_label
    correct = np.count_nonzero(matches)
    accuracy = float(correct)/float(test_img_num)
    print(i, "accuracy:", accuracy)
    
    #Display(test_img, test_label, test_h, test_w, 10, '1')    # 显示前10张图像及其真实标签
    #Display(test_img, result, test_h, test_w, 10, '2')    # 显示前10张图像及其预测标签

# 测试结果
# 1 accuracy: 0.9691
# 2 accuracy: 0.9627
# 3 accuracy: 0.9705
# 4 accuracy: 0.9682
# 5 accuracy: 0.9688
# 6 accuracy: 0.9677
# 7 accuracy: 0.9694
# 8 accuracy: 0.967
# 9 accuracy: 0.9659
# 10 accuracy: 0.9665
# 11 accuracy: 0.9668
# 12 accuracy: 0.9661
# 13 accuracy: 0.9653
# 14 accuracy: 0.964
# 15 accuracy: 0.9633
# 16 accuracy: 0.9632
# 17 accuracy: 0.963
# 18 accuracy: 0.9633
# 19 accuracy: 0.9632
# 20 accuracy: 0.9625