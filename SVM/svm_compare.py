import cv2
import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import time

# 1. 加载CSV文件
csv_file_path_1 = '0618.csv'
csv_file_path_2 = '0854.csv'

data_1 = pd.read_csv(csv_file_path_1)
data_2 = pd.read_csv(csv_file_path_2)

# 获取RGB值和标签
# X_1 = data_1[['R', 'G', 'B']].values
X_1 = data_1[['R', 'G']].values
y_1 = data_1['Label'].values
# X_2 = data_2[['R', 'G', 'B']].values
X_2 = data_2[['R', 'G']].values
y_2 = data_2['Label'].values

# 2. 训练SVM分类器
classifier_1_linear = svm.SVC(kernel='rbf', probability=True)
classifier_1_rbf = svm.SVC(kernel='rbf', probability=True)
classifier_1_poly = svm.SVC(kernel='poly', degree=3, coef0=1)
classifier_1_linear.fit(X_1, y_1)
classifier_1_rbf.fit(X_1, y_1)
classifier_1_poly.fit(X_1, y_1)

classifier_2_linear = svm.SVC(kernel='rbf', probability=True)
classifier_2_rbf = svm.SVC(kernel='rbf', probability=True)
classifier_2_poly = svm.SVC(kernel='poly', degree=3, coef0=1)
classifier_2_linear.fit(X_2, y_2)
classifier_2_rbf.fit(X_2, y_2)
classifier_2_poly.fit(X_2, y_2)

# 3. 加载图片并进行像素分类
img_path_1 = '0618.png'
img_path_2 = '0854.png'

img_1 = cv2.imread(img_path_1)
img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
img_2 = cv2.imread(img_path_2)
img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)

# 获取图像的宽高
h_1, w_1, _ = img_1.shape
h_2, w_2, _ = img_2.shape

# 创建一个新图像以存储分类结果
classified_img_1_linear = np.zeros((h_1, w_1), dtype=np.uint8)
classified_img_1_rbf = np.zeros((h_1, w_1), dtype=np.uint8)
classified_img_1_poly = np.zeros((h_1, w_1), dtype=np.uint8)
classified_img_2_linear = np.zeros((h_2, w_2), dtype=np.uint8)
classified_img_2_rbf = np.zeros((h_2, w_2), dtype=np.uint8)
classified_img_2_poly = np.zeros((h_2, w_2), dtype=np.uint8)

# 遍历每个像素进行分类
for i in range(h_1):
    for j in range(w_1):
        # pixel = img_1[i, j]
        pixel_1 = img_1[i, j][2]
        pixel_2 = img_1[i, j][1]
        # pixel = pixel.reshape(1, -1)
        pixel = [[pixel_1, pixel_2]]
        label_linear = classifier_1_linear.predict(pixel)
        classified_img_1_linear[i, j] = label_linear
        label_rbf = classifier_1_rbf.predict(pixel)
        classified_img_1_rbf[i, j] = label_rbf
        label_poly = classifier_1_poly.predict(pixel)
        classified_img_1_poly[i, j] = label_poly

# 遍历每个像素进行分类
for i in range(h_2):
    for j in range(w_2):
        # pixel = img_2[i, j]
        pixel_1 = img_2[i, j][2]
        pixel_2 = img_2[i, j][1]
        # pixel = pixel.reshape(1, -1)
        pixel = [[pixel_1, pixel_2]]
        label_linear = classifier_2_linear.predict(pixel)
        classified_img_2_linear[i, j] = label_linear
        label_rbf = classifier_2_rbf.predict(pixel)
        classified_img_2_rbf[i, j] = label_rbf
        label_poly = classifier_2_poly.predict(pixel)
        classified_img_2_poly[i, j] = label_poly

# 4. 可视化结果
# 展示原图和分类结果
plt.figure(figsize=(10, 5))

# 原图
plt.subplot(2, 4, 1)
plt.imshow(img_1)
plt.title('Original Image')
plt.axis('off')

# 分类结果
plt.subplot(2, 4, 2)
plt.imshow(classified_img_1_linear, cmap='gray')
plt.title('Linear Classified Image')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(classified_img_1_rbf, cmap='gray')
plt.title('RBF Classified Image')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(classified_img_1_poly, cmap='gray')
plt.title('POLY Classified Image')
plt.axis('off')

# 原图
plt.subplot(2, 4, 5)
plt.imshow(img_2)
plt.title('Original Image')
plt.axis('off')

# 分类结果
plt.subplot(2, 4, 6)
plt.imshow(classified_img_2_linear, cmap='gray')
plt.title('Linear Classified Image')
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(classified_img_2_rbf, cmap='gray')
plt.title('RBF Classified Image')
plt.axis('off')

plt.subplot(2, 4, 8)
plt.imshow(classified_img_2_poly, cmap='gray')
plt.title('POLY Classified Image')
plt.axis('off')

plt.show()
