import cv2
import numpy as np
import csv

class ChannelFeatureExtractor:
    def __init__(self, color_paths, binary_paths, output_csv):
        # 彩色图像路径列表
        self.color_paths = color_paths
        # 二值图像路径列表
        self.binary_paths = binary_paths
        # 输出 CSV 文件路径
        self.output_csv = output_csv

    # 读取图像文件
    def load_images(self):
        color_images = []
        binary_images = []

        for path in self.color_paths:
            # 读取彩色图像（BGR格式）
            color_img = cv2.imread(path)
            color_images.append(color_img)

        for path in self.binary_paths:
            # 读取二值图像（单通道）
            binary_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            binary_images.append(binary_img)

        return color_images, binary_images

    # RGB 转 HSV
    def rgb_to_hsv(self, rgb):
        return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]

    # 提取特征并保存为 CSV
    def extract_features_and_save(self, color_images, binary_images):
        # 打开 CSV 文件
        with open(self.output_csv, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['R', 'G', 'B', 'H', 'S', 'V', 'Label'])  # 写入表头

            for id in range(len(color_images)):
                height, width, _ = color_images[id].shape

                # 遍历每张彩色图像的每个像素
                for i in range(0, height, 10):
                    for j in range(0, width, 10):
                        # 提取 RGB 和 HSV 特征
                        rgb_values = color_images[id][i, j]  # 假设使用第一张彩色图像
                        b, g, r = rgb_values
                        h, s, v = self.rgb_to_hsv((r, g, b))

                        # 提取标签（0 或 255，使用对应的二值化图像）
                        binary_label = int(binary_images[id][i, j]/255)

                        # 写入 CSV 文件
                        csv_writer.writerow([r, g, b, h, s, v, binary_label])

    # 主函数：执行特征提取和保存
    def run(self):
        # 读取彩色图像和二值图像
        color_images, binary_images = self.load_images()

        # 提取特征并保存到 CSV 文件
        self.extract_features_and_save(color_images, binary_images)
        print(f"特征数据已保存为 {self.output_csv}")


# 主程序入口
if __name__ == "__main__":
    # 彩色图片和二值化图片的路径
    # color_paths = ['0618.png']
    # binary_paths = ['0618_truth.png']
    # color_paths = ['0854.png']
    # binary_paths = ['0854_truth.png']
    color_paths = ['1066.png']
    binary_paths = ['1066_truth.png']
    # 输出的CSV文件路径
    # output_csv = 'feature/channel/0618_channel_features.csv'
    # output_csv = 'feature/channel/0854_channel_features.csv'
    output_csv = 'feature/channel/1066_channel_features.csv'

    # 创建 ImageFeatureExtractor 实例并运行
    feature_extractor = ChannelFeatureExtractor(color_paths, binary_paths, output_csv)
    feature_extractor.run()
