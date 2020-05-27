#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 将输入的图片标准化

import math

import cv2
import numpy as np

from showImg import showImg

def getDistance(pointA, pointB):
    return math.sqrt((pointA[0]-pointB[0])**2+(pointA[1]-pointB[1])**2)

def drawExtension(points, img):
    d1 = getDistance(points[0], points[1])
    d2 = getDistance(points[1], points[2])
    if d1 < d2:
        Ax = (points[0, 0]+points[1, 0])/2
        Ay = (points[0, 1]+points[1, 1])/2
        Bx = (points[2, 0]+points[3, 0])/2
        By = (points[2, 1]+points[3, 1])/2
    else:
        Ax = (points[1, 0]+points[2, 0])/2
        Ay = (points[1, 1]+points[2, 1])/2
        Bx = (points[0, 0]+points[3, 0])/2
        By = (points[0, 1]+points[3, 1])/2
    #return [(int(Ax),int(Ay)), (int(Bx),int(By))]
    if Ay == By:
        endpoint = [(0, int(Ay)), (img.shape[1], int(Ay))]
    elif Ax == Bx:
        endpoint = [(int(Ax), 0), (int(Ax), img.shape[0])]
    else:
        endpoint = []
        k = (Ay-By)/(Ax-Bx)
        i = Ay-k*Ax
        if i >= 0 and i <= img.shape[0]:
            endpoint.append((0, int(i)))
        i = Ay+k*(img.shape[1]-Ax)
        if i >= 0 and i <= img.shape[0]:
            endpoint.append((img.shape[1], int(i)))
        k = (Ax-Bx)/(Ay-By)
        i = Ax-k*Ay
        if i >= 0 and i <= img.shape[1]:
            endpoint.append((int(i), 0))
        i = Ax+k*(img.shape[0]-Ay)
        if i >= 0 and i <= img.shape[1]:
            endpoint.append((int(i), img.shape[0]))
    cv2.line(img, endpoint[0], endpoint[1], 255, 4)
    #return ((endpoint[0][0]+endpoint[1][0])/2, (endpoint[0][1]+endpoint[1][1])/2)
     
def sobelFind(img_sobel, kernel, show):
    # 进行一次闭运算、一次开运算再进行两次闭运算
    # 闭运算连接断线，开运算防止过度粘连，闭运算确保连接
    close = cv2.morphologyEx(img_sobel, cv2.MORPH_CLOSE, kernel)
    open = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel)
    close2 = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel, iterations=1)
    if DEBUG:
        show.add(close, 'close', True)
        show.add(open, 'open', True)
        show.add(close2, 'close2', True)
    # 轮廓特征识别 滤除多余部分 保留长宽比足够大的部分
    draw = close2.copy()
    contours, _ = cv2.findContours(draw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        points = cv2.boxPoints(rect)
        d1 = getDistance(points[0], points[1])
        d2 = getDistance(points[1], points[2])
        if max(d1, d2)/min(d1, d2) < 25:
            cv2.drawContours(draw, [cnt], 0, 0, cv2.FILLED)
        else:
            #print(max(d1, d2)/min(d1, d2))
            cv2.drawContours(draw, [cnt], 0, 255, cv2.FILLED)
            lines.append([points, draw, cnt])
    # 先去除多余点再划线，防止出现断线
    for i in lines:
        drawExtension(i[0], i[1])
    # 重新获取轮廓 仅获取外轮廓
    contours, _ = cv2.findContours(draw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if DEBUG:
        img_contours = cv2.cvtColor(draw, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_contours, contours, -1, (0, 0, 255), 2)
        show.add(img_contours, 'cross_contours')
    # 检测识别到的线数量 去除多余线
    if len(contours) < 10:
        print('识别到的线数量小于10，错误！')
    elif len(contours) > 10 and len(contours) < 20:
        lines = []
        for cnt in contours:
            M = cv2.moments(cnt)
            lines.append([cnt, (M['m10'] / M['m00'], M['m01'] / M['m00'])])
        lines.sort(key=lambda x:x[1][0]+x[1][1])
        ptp = getDistance(lines[0][1], lines[-1][1])/4.5 #(.../9*2 最边缘两条线的距离/9得到的线距的2倍)
        for i in range(1, len(lines)-1):
            d1 = getDistance(lines[i][1], lines[i-1][1])
            d2 = getDistance(lines[i][1], lines[i+1][1])
            lines[i].append(d1+d2)
        lines[0].append(ptp)
        lines[-1].append(ptp)
        lines.sort(key=lambda x:abs(x[2]-ptp), reverse=True)
        for i in range(len(lines)-10):
            cv2.drawContours(draw, [lines[i][0]], 0, 0, cv2.FILLED)
    if DEBUG:
        show.add(draw, 'draw')
    return draw
    
def standardization(img_original, show=None):
    global DEBUG
    if show != None:
        DEBUG = True
    else:
        DEBUG = False
    # 中值滤波、高斯滤波
    img_blur = cv2.medianBlur(img_original, 1)
    img_blur = cv2.GaussianBlur(img_blur, (3, 3), 0)
    if DEBUG:
        show.add(img_blur, 'blur', True)
    
    # 亮度自动调节：减去闭运算后的图
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    close = cv2.morphologyEx(img_blur, cv2.MORPH_CLOSE, kernel)
    div = np.float32(img_blur) / close
    img_brightness_adjusted = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))
    if DEBUG:
        show.add(close, 'close', True)
        show.add(img_brightness_adjusted, 'brightness_adjusted', True)
    
    # 二次滤波
    img_blur2 = cv2.medianBlur(img_brightness_adjusted, 1)
    img_blur2 = cv2.GaussianBlur(img_blur2, (3, 3), 0)
    if DEBUG:
        show.add(img_blur2, 'blur2', True)
    
    # 二值化
    img_th = cv2.adaptiveThreshold(img_blur2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 10)
    if DEBUG:
        show.add(img_th, 'threshold', True)
    
    # 寻找轮廓
    contours, hierarchy = cv2.findContours(img_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if DEBUG:
        img_contours = cv2.cvtColor(img_th, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_contours, contours, -1, (0, 0, 255), 2)
        show.add(img_contours, 'contours', True)
    # 寻找面积最大的下一层级的轮廓数大于9的轮廓
    for cnt in sorted([[i, cv2.contourArea(contours[i])] for i in range(len(contours))], key=lambda x:x[1], reverse=True):
        if len([i for i in hierarchy[0] if i[3] == cnt[0]]) >= 9:
            mask = np.zeros(img_th.shape, np.uint8)
            cv2.drawContours(mask, contours, cnt[0], 255, cv2.FILLED)
            img_with_mask = cv2.bitwise_and(img_th, mask)
            break
    if 'img_with_mask' not in dir():
        exit(1)
    if DEBUG:
        show.add(img_with_mask, 'img_with_mask', True)
    
    # Sobel算子
    
    # x方向Sobel算子 找竖线
    dx = cv2.Sobel(img_with_mask, cv2.CV_16S, 1, 0)
    # 位深转化为 CV_8UC1
    dx = cv2.convertScaleAbs(dx)
    if DEBUG:
        show.add(dx, 'x', True)
    kernelx = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))
    dx = sobelFind(dx, kernelx, show)
    
    # y方向Sobel算子 找横线
    dy = cv2.Sobel(img_with_mask, cv2.CV_16S, 0, 1)
    # 位深转化为 CV_8UC1
    dy = cv2.convertScaleAbs(dy)
    if DEBUG:
        show.add(dy, 'y', True)
    kernely = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
    dy = sobelFind(dy, kernely, show)
    
    # 合并找交点
    img_cross = cv2.bitwise_and(dx, dy)
    img_cross = cv2.morphologyEx(img_cross, cv2.MORPH_CLOSE, kernel, iterations=1)
    if DEBUG:
        show.add(img_cross, 'cross')
    
    # 寻找每个交点的中心
    # 获取交点轮廓
    contours, _ = cv2.findContours(img_cross, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if DEBUG:
        img_contours = cv2.cvtColor(img_cross, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_contours, contours, -1, (0, 0, 255), 2)
        show.add(img_contours, 'cross_contours')
    # 通过图像矩获得质心
    if DEBUG:
        draw = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
    centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
            centroids.append((int(cx), int(cy)))
            if DEBUG:
                cv2.circle(draw, (int(cx), int(cy)), 3, (0, 0, 255), cv2.FILLED)
    if DEBUG:
        show.add(draw, 'draw')
    
    # 处理质心列表 按照位置分组排序
    centroids.sort(key=lambda x:x[1])
    centroids = [sorted(centroids[i:i+10], key=lambda x:x[0]) for i in range(0,100,10)]
    #print(centroids)
    
    # 进行透视变换
    img_puzzle = []
    for i in range(9):
        img_puzzle_y = []
        for j in range(9):
            pts1 = np.float32([centroids[i][j], centroids[i+1][j], centroids[i][j+1], centroids[i+1][j+1]])
            pts2 = np.float32([[0, 0], [0, 50], [50, 0], [50, 50]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(img_th, M, (50, 50))
            img_puzzle_y.append(dst)
            #show.add(dst, 'dst')
        img_puzzle.append(img_puzzle_y)
    if DEBUG:
        img_puzzle_show = []
        for i in img_puzzle:
            img_puzzle_show.append(np.hstack(i))
        img_puzzle_show = np.vstack(img_puzzle_show)
        show.add(img_puzzle_show, 'img_puzzle_show')
    return img_puzzle