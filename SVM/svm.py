import cv2
import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import time

# 1. 加载CSV文件
# csv_file_path = '0618.csv'
# csv_file_path = '0854.csv'
csv_file_path = '1066.csv'
data = pd.read_csv(csv_file_path)

# 获取RGB值和标签
# 特征：RGB值
X = data[['R', 'G', 'B']].values
# 标签
y = data['Label'].values

# 2. 训练SVM分类器
train_start = time.time()
classifier = svm.SVC(kernel='linear', probability=True)
# classifier = svm.SVC(kernel='rbf', probability=True)
# classifier = svm.SVC(kernel='poly', degree=3, coef0=1)
classifier.fit(X, y)
train_end = time.time()
train_time = train_end - train_start

# 3. 加载图片并进行像素分类
# img_path = '0618.png'
# img_path = '0854.png'
img_path = '1066.png'
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
        # 转换为2D数组
        pixel = pixel.reshape(1, -1)
        # 进行预测
        label = classifier.predict(pixel)
        # 存储分类结果
        classified_img[i, j] = label

test_end = time.time()
test_time = test_end - test_start
print("分类器训练时间为：%fs"%train_time)
print("分类器分类时间为：%fs"%test_time)

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
