#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#深度搜索算法解数独

class point(object):
    def __init__(self, x, y, available):
        self.x = x
        self.y = y
        self.available = available

#获取可填值
def getAvailableNum(x, y):
    availableNum = set(sudoku[y]).union(set([i[x] for i in sudoku]))
    for blockY in range(y//3*3, y//3*3+3):
        availableNum.update(set(sudoku[blockY][x//3*3:x//3*3+3]))
    return set(range(1, 10)).difference(availableNum)

# 深度搜索解数独
def solve(p):
    for avaNum in p.available:
        if avaNum in getAvailableNum(p.x, p.y):
            sudoku[p.y][p.x] = avaNum
            if len(searchList) == 0:
                return sudoku
            else:
                result = solve(searchList.pop())
                if result:
                    return result
    sudoku[p.y][p.x] = 0
    searchList.append(p)
    return None

# 初始化数据，获取每个值为零的方格的可填指，即去除横行、竖行、宫内的数字后的值，并生成列表
def solveSudoku(inputData):
    global sudoku
    global searchList
    sudoku = inputData
    searchList = []
    for y in range(len(sudoku)):
        for x in range(len(sudoku[y])):
            if sudoku[y][x] == 0:
                searchList.append(point(x, y, getAvailableNum(x, y)))
    return solve(searchList.pop())

if __name__ == '__main__':
    test = [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 9, 3, 6, 2, 8, 1, 4, 0],
                [0, 6, 0, 0, 0, 0, 0, 5, 0],
                [0, 3, 0, 0, 1, 0, 0, 9, 0],
                [0, 5, 0, 8, 0, 2, 0, 7, 0],
                [0, 4, 0, 0, 7, 0, 0, 6, 0],
                [0, 8, 0, 0, 0, 0, 0, 3, 0],
                [0, 1, 7, 5, 9, 3, 4, 2, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
    for row in solveSudoku(test):
        for col in row:
            print(col, end=' ')
        print('')