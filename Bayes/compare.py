import numpy as np
from matplotlib import pyplot as plt

from NaiveBayes import NavieBayes
from MultiBayes import MultiBayes
from sklearn import svm
import cv2
import pandas as pd

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
    truth_img = cv2.imread(ture_img_path, cv2.IMREAD_GRAYSCALE)

    data = pd.read_csv(csv_file_path)

    # 获取RGB值和标签
    # 特征
    X = data[['R', 'G', 'B']].values
    # 标签
    y = data['Label'].values

    bayes_navie = NavieBayes()
    bayes_navie.fit(X, y)
    img_navie = bayes_navie.train(img, 3)
    bayes_navie.compare(truth_img)

    bayes_multi = MultiBayes()
    bayes_multi.fit(X, y)
    img_multi = bayes_multi.train(img, 3)
    bayes_multi.compare(truth_img)

    classifier = svm.SVC(kernel='linear', probability=True)
    classifier.fit(X, y)
    h, w, _ = img.shape
    img_svm = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            # 获取像素的RGB值
            pixel = img[i, j]
            # 转换为2D数组
            pixel = pixel.reshape(1, -1)
            # 进行预测
            label = classifier.predict(pixel)
            # 存储分类结果
            if label == 0:
                label = 1
            else:
                label = 0
            img_svm[i, j] = label*255

    sum = np.sum(truth_img == img_svm)
    pro = sum / (truth_img.shape[0] * truth_img.shape[1])
    print("正确率为:", pro)


    plt.figure(figsize=(10, 5))

    # 原图
    plt.subplot(1, 3, 1)
    plt.imshow(img_navie, cmap='gray')
    plt.title('Navie Beyas Image')
    plt.axis('off')

    # 分类结果
    plt.subplot(1, 3, 2)
    plt.imshow(img_multi, cmap='gray')
    plt.title('Multi Beyas Image')
    plt.axis('off')

    # 分类结果
    plt.subplot(1, 3, 3)
    plt.imshow(img_svm, cmap='gray')
    plt.title('SVM Image')
    plt.axis('off')

    plt.show()