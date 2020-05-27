#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#使用matplotlib显示OpenCV的图片，用于debug

import cv2
from matplotlib import pyplot as plt


class showImg(object):
    def __init__(self):
        self.imgList = []
    
    def add(self, img, text, plot=False):
        self.imgList.append([img, text, plot])
    
    def save(self, path='./'):
        plotList = []
        for i in range(len(self.imgList)):
            if self.imgList[i][2]:
                plotList.append([str(i) + '. ' + self.imgList[i][1], self.imgList[i][0]])
            else:
                cv2.imwrite(path +str(i) + '_' + self.imgList[i][1] + '.png', self.imgList[i][0])
        plotList = [plotList[i:i+9] for i in range(0, len(plotList), 9)]
        for plot in range(len(plotList)):
            row = -(-len(plotList[plot])//3)
            plt.figure()
            num = 0
            for i in plotList[plot]:
                num += 1
                plt.subplot(row, 3, num)
                if i[1].ndim == 3:
                    plt.imshow(i[1][:, :, ::-1])
                else:
                    plt.imshow(i[1], cmap='gray')
                plt.title(i[0])
                plt.axis("off")
            plt.savefig(path + str(plot) + '_test.png')
            plt.close()