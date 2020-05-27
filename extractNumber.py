#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 提取输入的方形图片中的数字

import math

import cv2
import numpy as np

from showImg import showImg

def extractNumber(img_number, show, border = 1):
    if show != None:
        DEBUG = True
    else:
        DEBUG = False
    if img_number.shape[0] == img_number.shape[1]:
        length = img_number.shape[0]
        # 将图片内接圆外的所有像素设置为0
        for i in range(length):
            for j in range(length):
                if (i-length/2)**2+(j-length/2)**2 >= (length/2.52)**2:
                    img_number[i, j] = 0
        #print(cv2.countNonZero(img_number),70)
        # 非零像素数 大于 70
        if cv2.countNonZero(img_number) > 70:
            #return True
            # 找到外接矩形面积最大的轮廓
            contours, _ = cv2.findContours(img_number, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rects = []
            for cnt in contours:
                rects.append(cv2.boundingRect(cnt))
            x, y, w, h = sorted(rects, key=lambda i:i[2]*i[3])[-1]
            x, y, w, h = x-border, y-border, w+border*2, h+border*2
            # 最大外接矩形面积 大于 100
            if w*h > 100:
                # 矩形中心与图片中心的距离 大于 图片长度/4
                if (x+w/2-length/2)**2+(y+h/2-length/2)**2 < (length/4)**2:
                    img_number_roi = img_number[y:y+h, x:x+w]
                    if h > w:
                        img_return = np.zeros(shape=(h, h))
                        img_return[:, (h-w)//2:(h-w)//2+w] = img_number_roi
                    else:
                        img_return = np.zeros(shape=(w, w))
                        img_return[(w-h)//2:(w-h)//2+h, :] = img_number_roi
                    img_return = cv2.resize(img_return, (length, length))
                    if DEBUG:
                        show.add(img_return, 'img_return')
                    return True, img_return
    return False, None