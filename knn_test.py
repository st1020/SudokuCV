#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 临近算法(kNN)手写数字识别

from struct import unpack

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取训练集和测试集的图像和标签
train_img = np.load('train-images.npy')
train_label = np.load('train-labels.npy')
test_img = np.load('test-images.npy')
test_label = np.load('test-labels.npy')

train_img = train_img.reshape(train_img.shape[0], train_img.shape[1]*train_img.shape[2])
test_img = test_img.reshape(test_img.shape[0], test_img.shape[1]*test_img.shape[2])

# 将所有数据转成np.float32类型
train_img = train_img.astype(np.float32)
train_label = train_label.astype(np.float32)
test_img = test_img.astype(np.float32)
test_label = test_label.astype(np.float32)

# 调用OpenCV的knn实现分类
knn = cv2.ml.KNearest_create()
knn.train(train_img, cv2.ml.ROW_SAMPLE, train_label)

# 测试1到20所有K值并计算预测准确率
for i in range(1,21):
    ret, result, neighbours, dist = knn.findNearest(test_img, k = i)
    
    # 计算预测准确率
    matches = result == test_label
    correct = np.count_nonzero(matches)
    accuracy = float(correct)/float(len(test_img))
    print(i, "accuracy:", accuracy)

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