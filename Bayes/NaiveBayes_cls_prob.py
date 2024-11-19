import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, multivariate_normal
from scipy.stats import norm

class NaiveBayes():
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.road_pre = None
        self.not_road_pre = None
        self.road_data_m = None
        self.not_road_data_m = None
        self.road_data_cov = None
        self.not_road_data_cov = None
        self.classified_img = None
        self.flag = None
        self.means = {}
        self.stds = {}
        self.R_kde_0 = None
        self.G_kde_0 = None
        self.B_kde_0 = None
        self.R_kde_1 = None
        self.G_kde_1 = None
        self.B_kde_1 = None

        self.R_hist_0 = None
        self.R_bins_0 = None
        self.G_hist_0 = None
        self.G_bins_0 = None
        self.B_hist_0 = None
        self.B_bins_0 = None

        self.R_hist_1 = None
        self.R_bins_1 = None
        self.G_hist_1 = None
        self.G_bins_1 = None
        self.B_hist_1 = None
        self.B_bins_1 = None

    def prior_prob(self):
        # 计算先验概率
        self.road_pre = np.sum(self.y == 0)/len(self.y)
        self.not_road_pre = 1 - self.road_pre

    def fit(self, flag):
        self.flag = flag
        self.prior_prob()
        # 类别数量
        classes = np.unique(y)
        # 计算每个类的均值和方差
        for label in classes:
            # 获取类别 c 对应的数据
            X_label = X[y == label]
            if self.flag == 1:
                # 计算均值和协方差矩阵
                if label == 0:
                    self.road_data_m = np.mean(X_label, axis=0)
                    self.road_data_cov = np.cov(np.array(X_label), rowvar=False)
                elif label == 1:
                    self.not_road_data_m = np.mean(X_label, axis=0)
                    self.not_road_data_cov = np.cov(np.array(X_label), rowvar=False)

            elif self.flag == 2:
                # 计算每个通道的均值和标准差
                self.means[label] = np.mean(X_label, axis=0)
                self.stds[label] = np.std(X_label, axis=0)

            elif self.flag == 3:
                # 分别提取 RGB 通道数据
                R_class = X_label[:, 0]
                G_class = X_label[:, 1]
                B_class = X_label[:, 2]
                if label == 0:
                    self.R_kde_0 = gaussian_kde(R_class, bw_method=1.0)
                    self.G_kde_0 = gaussian_kde(G_class, bw_method=1.0)
                    self.B_kde_0 = gaussian_kde(B_class, bw_method=1.0)
                elif label == 1:
                    self.R_kde_1 = gaussian_kde(R_class, bw_method=1.0)
                    self.G_kde_1 = gaussian_kde(G_class, bw_method=1.0)
                    self.B_kde_1 = gaussian_kde(B_class, bw_method=1.0)

            elif self.flag == 4:
                R_class = X_label[:, 0]
                G_class = X_label[:, 1]
                B_class = X_label[:, 2]
                if label == 0:
                    # 计算每个通道的条件概率直方图
                    self.R_hist_0, self.R_bins_0 = np.histogram(R_class, bins=256, range=(0, 256), density=True)
                    self.G_hist_0, self.G_bins_0 = np.histogram(G_class, bins=256, range=(0, 256), density=True)
                    self.B_hist_0, self.B_bins_0 = np.histogram(B_class, bins=256, range=(0, 256), density=True)
                elif label == 1:
                    # 计算每个通道的条件概率直方图
                    self.R_hist_1, self.R_bins_1 = np.histogram(R_class, bins=256, range=(0, 256), density=True)
                    self.G_hist_1, self.G_bins_1 = np.histogram(G_class, bins=256, range=(0, 256), density=True)
                    self.B_hist_1, self.B_bins_1 = np.histogram(B_class, bins=256, range=(0, 256), density=True)


    def train(self, img, n):
        step = n//2
        # 获取图像的宽高
        h, w, _ = img.shape
        # 创建一个新图像以存储分类结果
        self.classified_img = np.zeros((h, w), dtype=np.uint8)
        for i in range(step, h - step):
            for j in range(step, w - step):
                region = np.mean(img[i - step:i + step + 1, j - step:j + step + 1, :], axis=(0, 1))
                # 计算区域中每个像素的后验概率
                if self.flag == 1:
                    prob_pos = self.road_pre * multivariate_normal.pdf(region, mean=self.road_data_m,
                                                                       cov=self.road_data_cov)
                    prob_neg = self.not_road_pre * multivariate_normal.pdf(region, mean=self.not_road_data_m,
                                                                           cov=self.not_road_data_cov)
                else:
                    prob_pos = self.road_pre * self.conditional_probability(region[0], region[1], region[2], 0, self.flag)
                    prob_neg = self.not_road_pre * self.conditional_probability(region[0], region[1], region[2], 1, self.flag)

                if prob_pos >= prob_neg:
                    self.classified_img[i, j] = 255
                else:
                    self.classified_img[i, j] = 0

        # # 显示结果图像
        # cv2.imshow('Result', self.classified_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return self.classified_img

    def conditional_probability(self, R, G, B, label, flag):
        if flag == 2:
            mean_r, mean_g, mean_b = self.means[label]
            std_r, std_g, std_b = self.stds[label]

            # 计算每个通道的高斯概率密度
            p_r = norm.pdf(R, mean_r, std_r)
            p_g = norm.pdf(G, mean_g, std_g)
            p_b = norm.pdf(B, mean_b, std_b)

        # 核密度估计 RGB 通道的概率密度
        elif flag == 3:
            if label == 0:
                p_r = self.R_kde_0(R)
                p_g = self.G_kde_0(G)
                p_b = self.B_kde_0(B)
            elif label == 1:
                p_r = self.R_kde_1(R)
                p_g = self.G_kde_1(G)
                p_b = self.B_kde_1(B)
        elif flag == 4:
            if label == 0:
                # 计算每个 RGB 通道对应 RGB 值的条件概率
                p_r = self.R_hist_0[np.digitize(R, self.R_bins_0) - 1]
                p_g = self.G_hist_0[np.digitize(G, self.G_bins_0) - 1]
                p_b = self.B_hist_0[np.digitize(B, self.B_bins_0) - 1]
            elif label == 1:
                # 计算每个 RGB 通道对应 RGB 值的条件概率
                p_r = self.R_hist_1[np.digitize(R, self.R_bins_1) - 1]
                p_g = self.G_hist_1[np.digitize(G, self.G_bins_1) - 1]
                p_b = self.B_hist_1[np.digitize(B, self.B_bins_1) - 1]

        # 返回 RGB 通道的联合概率
        return p_r * p_g * p_b

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

    img_path = '0854.png'
    ture_img_path = '0854_truth.png'
    csv_file_path = '0854.csv'

    # img_path = '1066.png'
    # ture_img_path = '1066_truth.png'
    # csv_file_path = '1066.csv'

    img = cv2.imread(img_path)
    # 转换为RGB格式
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    data = pd.read_csv(csv_file_path)

    # 获取RGB值和标签
    # 特征，归一化
    X = data[['R', 'G', 'B']].values
    # 标签
    y = data['Label'].values

    truth_img = cv2.imread(ture_img_path, cv2.IMREAD_GRAYSCALE)

    bayes = NaiveBayes(X, y)

    result = [np.zeros((250, 500), dtype=np.uint8) for i in range(4)]

    for i in range(1, 5):
        bayes.fit(flag = i)

        result[i - 1] = bayes.train(img, 3)

        truth_img = cv2.imread(ture_img_path, cv2.IMREAD_GRAYSCALE)

        # bayes.post_process()

        bayes.compare(truth_img)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(result[0], cmap='gray')
    plt.title('Multivariate normal distribution')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(result[1], cmap='gray')
    plt.title('Gaussian normal estimation')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(result[2], cmap='gray')
    plt.title('Kernel density estimation')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(result[3], cmap='gray')
    plt.title('Histogram estimation')
    plt.axis('off')

    plt.show()
