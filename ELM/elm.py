import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import time

class ELM:
    def __init__(self, n_input, n_hidden):
        # 输入特征数量
        self.n_input = n_input
        # 隐藏层节点数量
        self.n_hidden = n_hidden
        # 输入层到隐藏层的权重
        # self.W = np.random.uniform(0, 1.0, size=(self.n_input, self.n_hidden))
        self.W = np.random.rand(n_input, n_hidden)
        # 隐藏层偏置
        self.B = np.random.rand(n_hidden)
        # self.B = np.random.uniform(-0.4, 0.4, size=(1, self.n_hidden))

    def activation_function(self, X):
        # Sigmoid激活函数
        return 1 / (1 + np.exp(-X))

    def fit(self, X, y):
        # 独热编码器
        encoder = OneHotEncoder()
        # 将输入的T转换为独热编码的形式
        y = encoder.fit_transform(y.reshape(-1, 1)).toarray()
        # 计算隐藏层输出矩阵
        H = self.activation_function(np.dot(X, self.W) + self.B)
        # 计算输出权重
        self.beta = np.linalg.pinv(H) @ y

    def predict(self, X):
        H = self.activation_function(np.dot(X, self.W) + self.B)
        res = np.dot(H, self.beta)
        # print(res)
        return np.argmax(res, axis=1)

# 使用示例
if __name__ == "__main__":
    # 1. 加载CSV文件
    # csv_file_path = '0618.csv'
    csv_file_path = '0854.csv'
    # csv_file_path = '1066.csv'
    # img_path = '0618.png'
    img_path = '0854.png'
    # img_path = '1066.png'
    data = pd.read_csv(csv_file_path)

    # 获取RGB值和标签
    # 特征：RGB值
    X = data[['R', 'G', 'B']].values
    X_train = []
    for point in X:
        temp = []
        for i in point:
            temp.append(i/255)
        X_train.append(temp)
    # 标签
    y = data['Label'].values

    # 2. 训练SVM分类器
    train_start = time.time()
    n_input = X.shape[1]
    n_hidden = 10
    # n_hidden = 10
    # n_hidden = 20
    elm = ELM(n_input, n_hidden)
    elm.fit(X_train, y)
    train_end = time.time()
    train_time = train_end - train_start

    # 3. 加载图片并进行像素分类
    img = cv2.imread(img_path)
    # 转换为RGB格式
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 获取图像的宽高
    h, w, _ = img.shape

    # 创建一个新图像以存储分类结果
    classified_img = np.zeros((h, w), dtype=np.uint8)
    test_start = time.time()
    # 遍历每个像素进行分类
    for i in range(h):
        for j in range(w):
            # 获取像素的RGB值
            pixel = img[i, j]
            temp = np.array([pixel[0]/255, pixel[1]/255, pixel[2]/255])
            pixel = temp.reshape(1, -1)
            # 进行预测
            label = elm.predict(pixel)
            # 存储分类结果
            # if label == 0:
            if label == 0 or label == 1:
                classified_img[i, j] = 1
            else:
                classified_img[i, j] = 0


    test_end = time.time()
    test_time = test_end - test_start
    print("分类器训练时间为：%fs" % train_time)
    print("分类器分类时间为：%fs" % test_time)

    # 均值滤波
    # result = cv2.blur(classified_img, (11, 11))
    # 中值滤波
    result = cv2.medianBlur(classified_img, 11)

    # 4. 可视化结果
    # 展示原图和分类结果
    plt.figure(figsize=(10, 5))

    # 原图
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    # 分类结果
    plt.subplot(1, 3, 2)
    plt.imshow(classified_img, cmap='gray')
    plt.title('Classified Image')
    plt.axis('off')

    # 滤波结果
    plt.subplot(1, 3, 3)
    plt.imshow(result, cmap='gray')
    plt.title('medianBlur with (11) Image')
    plt.axis('off')

    plt.show()
