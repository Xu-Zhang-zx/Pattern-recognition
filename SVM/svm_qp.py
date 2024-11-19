import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

class SVM:
    def __init__(self, X, y, C=1.0, learning_rate=0.01, max_iter=1000):
        self.X = X
        self.y = y
        self.C = C
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.show_flag = 0

    def fit(self):
        self.show_flag = 1
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        m = X.shape[0]

        for _ in range(self.max_iter):
            hinge_loss = 0
            dw = self.w.copy()
            db = 0
            for i in range(m):
                # 计算 SVM 损失的梯度
                if self.y[i] * (np.dot(self.X[i], self.w) + self.b) < 1:
                    hinge_loss += 1 - self.y[i] * (np.dot(self.X[i], self.w) + self.b)
                    dw -= self.learning_rate * self.C * self.y[i] * self.X[i]
                    db -= self.learning_rate * self.C * self.y[i]

            if hinge_loss == 0:
                print("提前结束!")
                break

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
        print(self.w, self.b)

    def predict(self, X):
        return np.sign(X @ self.w + self.b)

    def show(self):
        if not self.show_flag:
            print("抱歉,你还没有进行SVM训练!")
            return 0
        else:
            # 生成x和y的值
            x = np.linspace(0, 255, 100)
            y = np.linspace(0, 255, 100)

            # 创建网格
            X, Y = np.meshgrid(x, y)

            # 计算z的值
            Z = -(self.w[0] / self.w[2]) * X - (self.w[1] / self.w[2]) * Y - (self.b / self.w[2])

            # 绘制三维平面
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Z, color="r",alpha=0.5, rstride=100, cstride=100)

            # 绘制离散点
            for num in range(len(self.X)):
                if self.y[num] == 1:
                    ax.scatter(self.X[num][0], self.X[num][1], self.X[num][2], color='b', marker='o', s=100, label='Doad')
                else:
                    ax.scatter(self.X[num][0], self.X[num][1], self.X[num][2], color='y', marker='o', s=100, label='Not Doad')

            # 设置标签
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')

            # 显示图形
            plt.show()


# 示例数据
if __name__ == "__main__":

    # 1. 加载CSV文件
    # csv_file_path = '0618.csv'
    # csv_file_path = '0854.csv'
    csv_file_path = '1066.csv'
    data = pd.read_csv(csv_file_path)

    # 获取RGB值和标签
    X = data[['R', 'G', 'B']].values  # 特征：RGB值
    y = data['Label'].values  # 标签

    # 训练 SVM 模型
    train_start = time.time()
    # svm = SVM(X, y, C = 1.0, learning_rate = 0.2, max_iter = 10000)
    # svm = SVM(X, y, C = 1.0, learning_rate = 0.2, max_iter = 10000)
    # svm = SVM(X, y, C = 1.0, learning_rate = 0.005, max_iter = 100000)
    svm = SVM(X, y, C = 1.0, learning_rate = 0.002, max_iter = 100000)
    # svm = SVM(X, y, C = 1.0, learning_rate = 0.01, max_iter = 100000)
    svm.fit()
    train_end = time.time()
    train_time = train_end - train_start
    # 展示分类效果
    svm.show()

    # 3. 加载图片并进行像素分类
    # img_path = '0618.png'
    # img_path = '0854.png'
    img_path = '1066.png'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

    # 获取图像的宽高
    h, w, _ = img.shape

    # 创建一个新图像以存储分类结果
    classified_img = np.zeros((h, w), dtype=np.uint8)

    test_start = time.time()
    # 遍历每个像素进行分类
    for i in range(h):
        for j in range(w):
            pixel = img[i, j]  # 获取像素的RGB值
            pixel = pixel.reshape(1, -1)  # 转换为2D数组
            label = svm.predict(pixel)  # 进行预测
            classified_img[i, j] = label  # 存储分类结果
    test_end = time.time()
    test_time = test_end - test_start
    print("分类器训练时间为：%fs" % train_time)
    print("分类器分类时间为：%fs" % test_time)

    # 4. 可视化结果
    # 展示原图和分类结果
    plt.figure(figsize=(10, 5))

    # 原图
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    # 分类结果
    plt.subplot(1, 2, 2)
    plt.imshow(classified_img, cmap='gray')
    plt.title('Classified Image')
    plt.axis('off')

    plt.show()

