import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import time

class MultiBayes():
    def __init__(self):
        self.road_pre = None
        self.not_road_pre = None
        self.road_data_m = None
        self.not_road_data_m = None
        self.road_data_cov = None
        self.not_road_data_cov = None
        self.classified_img = None

    def fit(self, X, y):
        count = len(X)
        road_data = []
        not_road_data = []
        num = 0
        for i in range(count):
            if y[i] == 0:
                road_data.append(X[i])
                num += 1
            elif y[i] == 1:
                not_road_data.append(X[i])
        # 计算先验概率
        self.road_pre = num / count
        self.not_road_pre = 1 - self.road_pre
        # 计算均值
        self.road_data_m = np.mean(road_data, axis = 0)
        self.not_road_data_m = np.mean(not_road_data, axis = 0)
        # 计算协方差矩阵
        self.road_data_cov = np.cov(np.array(road_data), rowvar = False)
        self.not_road_data_cov = np.cov(np.array(not_road_data), rowvar = False)

    def train(self, img, n):
        step = n//2
        # 获取图像的宽高
        h, w, _ = img.shape
        # 创建一个新图像以存储分类结果
        self.classified_img = np.zeros((h, w), dtype=np.uint8)
        for i in range(step, h - step):
            for j in range(step, w - step):
                region = img[i - step:i + step + 1, j - step:j + step + 1, :]
                # 计算区域中每个像素的类条件概率
                prob_pos = multivariate_normal.pdf(region.reshape(-1, 3), mean=self.road_data_m,
                                                              cov=self.road_data_cov)
                prob_neg = multivariate_normal.pdf(region.reshape(-1, 3), mean=self.not_road_data_m,
                                                              cov=self.not_road_data_cov)

                # print("**********************",self.road_pre, self.not_road_pre,"**************************")

                prob_road = self.road_pre
                prob_not_road = self.not_road_pre

                for cnt in range(n**2):
                    prob_road = prob_pos[cnt] * prob_road / (prob_pos[n] * prob_road + prob_neg[n] * (1 - prob_road))
                    prob_not_road = prob_neg[cnt] * prob_not_road / (prob_neg[n] * prob_not_road + prob_pos[n] * (1 - prob_not_road))

                if prob_road > prob_not_road:
                    self.classified_img[i, j] = 255
                else:
                    self.classified_img[i, j] = 0

        # # 显示结果图像
        # cv2.imshow('Result', self.classified_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return self.classified_img

    def post_process(self):
        # 定义形态学操作的结构元素（kernel），可以改变形态学操作的效果
        kernel = np.ones((11, 11), np.uint8)
        # 形态学操作 - 闭运算（Closing），先膨胀再腐蚀
        closed = cv2.morphologyEx(self.classified_img, cv2.MORPH_CLOSE, kernel)

        # 显示结果
        plt.figure(figsize=(10, 7))

        plt.subplot(1, 2, 1)
        plt.title('Original Binary Image')
        plt.imshow(self.classified_img, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Closed Image')
        plt.imshow(closed, cmap='gray')
        plt.axis('off')

        plt.show()

    def compare(self, gruth):
        sum = np.sum(gruth == self.classified_img)
        pro = sum/(gruth.shape[0] * gruth.shape[1])
        print("正确率为:", pro)
        return 1

if __name__ == "__main__":
    # 1. 加载文件
    # img_path = '0618.png'
    # ture_img_path = '0618_truth.png'
    # csv_file_path = '0618.csv'

    # img_path = '0854.png'
    # ture_img_path = '0854_truth.png'
    # csv_file_path = '0854.csv'

    img_path = '1066.png'
    ture_img_path = '1066_truth.png'
    csv_file_path = '1066.csv'

    img = cv2.imread(img_path)
    # 转换为RGB格式
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    data = pd.read_csv(csv_file_path)

    # 获取RGB值和标签
    # 特征
    X = data[['R', 'G', 'B']].values
    # 标签
    y = data['Label'].values

    bayes = MultiBayes()

    bayes.fit(X, y)

    bayes.train(img, 3)

    truth_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    bayes.post_process()

    bayes.compare(truth_img)
