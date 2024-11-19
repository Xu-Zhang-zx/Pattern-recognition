import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import csv


class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()

        # 卷积层和池化层设计
        # 输入通道3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化层，2x2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # 假设输入的图像尺寸为25x25，经过3次池化后，尺寸会变为 3x3
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ImageDataset(data.Dataset):
    def __init__(self, image_folder, labels_file, transform = None):
        """
        自定义数据集类，加载图像和标签
        :param image_folder: 存储图像的文件夹路径
        :param labels_file: 存储标签的文件路径
        :param transform: 任何需要的转换（如标准化等）
        """
        self.image_folder = image_folder
        self.labels_file = labels_file
        self.transform = transform

        # 获取图像文件和标签
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]
        # 确保文件顺序一致
        self.image_files.sort()
        self.labels = self.load_labels()

    def load_labels(self):
        """读取标签文件并返回标签列表"""
        with open(self.labels_file, 'r') as f:
            labels = [int(line.strip()) for line in f.readlines()]
        return labels

    def __len__(self):
        """返回数据集大小"""
        return len(self.image_files)

    def __getitem__(self, idx):
        """返回指定索引的图像和标签"""
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_name)

        img = cv2.imread(img_path)
        # 确保图像为 25x25
        img = cv2.resize(img, (25, 25))

        if self.transform:
            # 应用任何转换
            img = self.transform(img)

        label = self.labels[idx]
        return img, label


def save_to_csv(features, labels, output_csv):
    """将特征和标签保存到CSV文件"""
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([f'Feature_{i + 1}' for i in range(features.shape[1])] + ['Label'])
        for feature, label in zip(features, labels):
            csv_writer.writerow(list(feature) + [label])


def extract_features(image_folder, labels_file, output_csv):
    """从图像文件夹中提取特征并保存到CSV"""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = ImageDataset(image_folder, labels_file, transform)
    dataloader = data.DataLoader(dataset, batch_size = 32, shuffle = False)

    model = CNNFeatureExtractor()

    model.eval()

    all_features = []
    all_labels = []

    # 设置 TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir='logs')

    # 这里我们假设 batch_size 是 32，选择一个虚拟的输入张量来初始化图形
    dummy_input = torch.randn(1, 3, 25, 25)
    writer.add_graph(model, dummy_input)

    with torch.no_grad():
        for images, labels in dataloader:
            # 提取特征
            features = model(images)
            all_features.append(features.numpy())
            all_labels.extend(labels.numpy())

    # 合并所有批次的特征
    all_features = np.concatenate(all_features, axis =  0)

    # 保存特征和标签到CSV文件
    save_to_csv(all_features, all_labels, output_csv)
    print(f"特征和标签已保存到 {output_csv}")

# 使用示例
if __name__ == "__main__":

    image_folder = "dataset/data_0618"
    labels_file = "dataset/data_0618/labels.txt"
    output_csv = "feature/cnn/0618_cnn_features.csv"
    # image_folder = "dataset/data_0854"
    # labels_file = "dataset/data_0854/labels.txt"
    # output_csv = "feature/cnn/0854_cnn_features.csv"
    # image_folder = "dataset/data_1066"
    # labels_file = "dataset/data_1066/labels.txt"
    # output_csv = "feature/cnn/1066_cnn_features.csv"

    # 提取特征并保存到CSV
    extract_features(image_folder, labels_file, output_csv)


