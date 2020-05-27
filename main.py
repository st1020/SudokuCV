#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#机器视觉识别数独题目并使用深度搜索解数独

import os
import math
import shutil

import cv2
import numpy as np

from sudoku import solveSudoku
from showImg import showImg
from standardization import standardization
from extractNumber import extractNumber
from ocr import ocr

DEBUG = False

ALL = False
#ALL = True

def main(img_file):
    if DEBUG:
        file = os.path.basename(img_file)
        if os.path.exists('./test/' + file + '/'):
            shutil.rmtree('./test/' + file + '/')
        os.makedirs('./test/' + file + '/')
        show = showImg()
    else:
        show = None
    
    # 载入灰度图
    img_original = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    if DEBUG:
        print(img_file, img_original.shape)
        show.add(img_original, 'original', True)
    
    # 对图片进行标准化 得到一个 9x9 列表 对应每个格子的图片
    standard_imgs = standardization(img_original, show)
    
    # 遍历图片列表 提取数字
    number_imgs = []
    for i in range(9):
        for j in range(9):
            is_number, img_number = extractNumber(standard_imgs[i][j], show)
            if is_number:
                number_imgs.append((img_number, i , j))
    
    # OCR数字识别
    # 1 kNN识别(Mnist 数据集)：准确率较低
    # 2 kNN识别(经过处理的 Mnist 数据集)：准确率较高速度较快(推荐)
    # 3 tesseract-ocr 单字识别：识别速度较慢并可能出现缺字
    # 4 tesseract-ocr 合并识别：识别准确率较高但也会出现缺字
    # 5 自动识别：先进行4，若缺字则进行2和1补全 优先级：4>3>1 若4缺字过多会导致运算量随缺字数呈指数上升，造成内存溢出
    # 其中2应配合extractNumber函数的border参数设为1，其余设为5
    orc_success, number = ocr([x[0] for x in number_imgs], 2)
    if not orc_success:
        print('OCR识别失败')
        os.exit()
    
    print('方格数：' + str(len(number)))
    
    # 处理识别到的数字
    number = [(number[i], number_imgs[i][1], number_imgs[i][2]) for i in range(len(number))]
    sudoku = [[0]*9 for _ in range(9)]
    for n, i, j in number:
        sudoku[i][j] = n
    
    # 显示初始状态
    for row in sudoku:
        for col in row:
            print(col, end=' ')
        print('')
    
    # 深度搜索解数独并显示
    sudokuResult = solveSudoku(sudoku)
    if sudokuResult != None:
        print('\n数独的解：')
        for row in sudokuResult:
            for col in row:
                print(col, end=' ')
            print('')
    else:
        print('\n无解！')
    
    if DEBUG:
        show.save('./test/' + file + '/')
    
if __name__ == '__main__':
    if ALL:
        for files in sorted(os.listdir('images')):
            main('./images/' + files)
    else:
        files = 'h2.png'
        main('./images/' + files)