import os
import cv2
import numpy as np


def save_image_patches(image, binary_mask, n, dataset_folder, index_start):
    """
    将图像分割成n×n大小的小图像，并根据二值化图像计算标签
    :param image: 输入的RGB彩色图像
    :param binary_mask: 二值化图像，黑白区域
    :param n: 小块的尺寸
    :param dataset_folder: 数据集文件夹路径
    :param index_start: 当前图片开始的索引
    :return: 更新后的索引值
    """
    # 获取图像的高度和宽度
    height, width, _ = image.shape
    mask_height, mask_width = binary_mask.shape

    # 确保二值化图像大小与彩色图像一致
    assert height == mask_height and width == mask_width, "彩色图像和二值化图像的大小必须相同"

    # 创建文件夹用于保存图片
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # 标签文件路径
    label_file = os.path.join(dataset_folder, 'labels.txt')

    # 开始保存图片和标签
    label_list = []
    index = index_start  # 图片索引，从给定的开始索引开始

    for i in range(0, height, n):
        for j in range(0, width, n):
            # 切分n×n的小块
            patch = image[i:i + n, j:j + n]
            mask_patch = binary_mask[i:i + n, j:j + n]

            # 计算该小块的白色像素数量
            white_pixels = np.sum(mask_patch == 255)
            total_pixels = n * n

            # 根据白色像素数量来决定标签
            # 如果白色像素超过50%的比例，则视为道路，标签为1，否则为非道路，标签为0
            if white_pixels > total_pixels / 2:
                label = 1  # 道路
            else:
                label = 0  # 非道路

            # 保存图片
            patch_filename = f'{index:03d}.png'
            patch_path = os.path.join(dataset_folder, patch_filename)
            cv2.imwrite(patch_path, patch)

            # 保存标签
            label_list.append(str(label))

            index += 1

    # 将所有标签写入txt文件
    with open(label_file, 'a') as f:  # 使用'a'模式追加内容
        f.write("\n".join(label_list) + "\n")

    return index  # 返回更新后的索引


def main():
    # 设置n×n小块的尺寸
    n = 25

    # 指定保存的dataset文件夹路径
    dataset_folder = 'dataset/data_all'

    # 三对图片和二值化图像的文件名
    images = ['0618.png', '0854.png', '1066.png']
    binary_masks = ['0618_truth.png', '0854_truth.png', '1066_truth.png']

    # 初始化图片索引
    index_start = 0

    for image_filename, mask_filename in zip(images, binary_masks):
        # 读取RGB彩色图像和二值化图像
        image = cv2.imread(image_filename)
        binary_mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)

        # 执行图像划分和标签生成
        index_start = save_image_patches(image, binary_mask, n, dataset_folder, index_start)

    print(f"数据集保存成功！共保存了 {index_start} 张小图片。")


if __name__ == "__main__":
    main()
