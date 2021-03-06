# SudokuCV
基于OpenCV的数独图片处理和求解
使用的软件版本：
- Python 3.8
- OpenCV 4.2


数字识别数据集：
[Mnist](http://yann.lecun.com/exdb/mnist/)


参考项目：
[sudoku_opencv_py](https://github.com/Howlclat/sudoku_opencv_py)


实现流程：
1. 对图片进行标准化，得到单个方格的二值化图片
2. 识别方格内是否有数字
3. 使用 kNN 算法 或 Tesseract OCR 进行数字识别，得到数独数组
4. 使用 深度搜索算法 进行数独求解

（详情可以参看文件内注释）


已知问题：
1. 当前对图片标准化算法处理时间较长，并且为了提高对于数独方格均为直线的图片的识别成功率，降低了对于方格存在扭曲的图片的识别成功率（如纸张皱褶）。
2. 数字识别（OCR）的识别成功率较低，提供的成功率较高的几个方案准确率也仅约 90% 。可使用更加先进的机器学习算法进行训练、训练自己的数据集或使用网络OCR提高识别准确率。
