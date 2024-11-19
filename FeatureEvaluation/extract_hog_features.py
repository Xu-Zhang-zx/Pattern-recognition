import cv2
import numpy as np
import os
from skimage.feature import hog
import csv


class HOGFeatureExtractor:
    def __init__(self, image_folder, labels_file, output_csv):
        """
        初始化类
        :param image_folder: 存储图像的文件夹路径
        :param labels_file: 存储标签的文本文件路径
        :param output_csv: 输出CSV文件的路径
        """
        self.image_folder = image_folder
        self.labels_file = labels_file
        self.output_csv = output_csv

    def load_images(self):
        """
        从文件夹中加载所有图像，假设所有图像为25×25大小
        :return: 图像文件列表
        """
        image_files = [f for f in os.listdir(self.image_folder) if f.endswith('.jpg') or f.endswith('.png')]
        # 排序确保图像顺序一致
        image_files.sort()
        images = []

        for img_file in image_files:
            img_path = os.path.join(self.image_folder, img_file)
            # 以灰度图方式读取
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # 确保图像为25x25大小
            if img is not None and img.shape == (25, 25):
                images.append(img)
            else:
                print(f"Warning: Image {img_file} is not 25x25 or is invalid.")
        return images

    def load_labels(self):
        """
        读取标签文件，并返回标签列表
        :return: 标签列表
        """
        labels = []

        with open(self.labels_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                # 读取每行并转换为整数
                labels.append(int(line.strip()))
        return labels

    def extract_hog_features(self, images):
        """
        提取图像的HOG特征
        :param images: 图像列表
        :return: HOG特征列表
        """
        hog_features = []

        for img in images:
            # 提取HOG特征
            feature, _ = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
            hog_features.append(feature)

        return hog_features

    def save_to_csv(self, features, labels):
        """
        将HOG特征和标签保存到CSV文件
        :param features: HOG特征列表
        :param labels: 标签列表
        """
        with open(self.output_csv, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([f'Feature_{i + 1}' for i in range(len(features[0]))] + ['Label'])  # 写入表头
            for feature, label in zip(features, labels):
                # 写入每一行数据
                csv_writer.writerow(list(feature) + [label])

    def run(self):
        """
        主程序执行
        1. 读取图像
        2. 读取标签
        3. 提取HOG特征
        4. 保存到CSV
        """
        # 读取图像和标签
        images = self.load_images()
        labels = self.load_labels()

        if len(images) != len(labels):
            raise ValueError("图像数量与标签数量不一致！")

        # 提取HOG特征
        hog_features = self.extract_hog_features(images)

        # 保存特征到CSV文件
        self.save_to_csv(hog_features, labels)
        print(f"HOG特征和标签已保存到 {self.output_csv}")


# 使用示例
if __name__ == "__main__":
    # image_folder = "dataset/data_0618"
    # labels_file = "dataset/data_0618/labels.txt"
    # output_csv = "feature/hog/0618_hog_features.csv"
    # image_folder = "dataset/data_0854"
    # labels_file = "dataset/data_0854/labels.txt"
    # output_csv = "feature/hog/0854_hog_features.csv"
    image_folder = "dataset/data_1066"
    labels_file = "dataset/data_1066/labels.txt"
    output_csv = "feature/hog/1066_hog_features.csv"
    # 创建HOGFeatureExtractor对象并运行
    feature_extractor = HOGFeatureExtractor(image_folder, labels_file, output_csv)
    feature_extractor.run()
