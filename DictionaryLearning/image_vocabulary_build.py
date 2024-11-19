import os
import cv2

def split_image(image, n, flag):
    """将图像分割成n×n大小的小图像"""
    if flag == 1:
        height, width, _ = image.shape
    else:
        height, width = image.shape
    patches = []
    for y in range(0, height, n):
        for x in range(0, width, n):
            patch = image[y:y + n, x:x + n]
            # 确保最后的patch是n×n大小
            if patch.shape[0] == n and patch.shape[1] == n:
                patches.append(patch)
    return patches


def save_patches(patches, folder_path, start_index=0):
    """将小图像保存到指定的文件夹"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i, patch in enumerate(patches):
        file_name = f"{start_index + i:03d}.png"  # 按顺序命名文件
        file_path = os.path.join(folder_path, file_name)
        cv2.imwrite(file_path, patch)


def main():
    # 图片路径设置
    # color_images = ['0618.png', '0854.png', '1066.png']
    # binary_images = ['0618_truth.png', '0854_truth.png', '1066_truth.png']
    color_images = ['0618.png']
    binary_images = ['0618_truth.png']

    # color_images = ['0854.png']
    # binary_images = ['0854_truth.png']
    #
    # color_images = ['1066.png']
    # binary_images = ['1066_truth.png']

    # 设置n×n的大小
    n = 5

    # 创建保存小图像的文件夹
    # original_folder = 'dataset/image_vocabulary_build/origin'
    # split_folder = 'dataset/image_vocabulary_build/split'
    original_folder = 'dataset/image_vocabulary_build/0618/origin'
    split_folder = 'dataset/image_vocabulary_build/0618/split'
    # original_folder = 'dataset/image_vocabulary_build/0854/origin'
    # split_folder = 'dataset/image_vocabulary_build/0854/split'
    # original_folder = 'dataset/image_vocabulary_build/1066/origin'
    # split_folder = 'dataset/image_vocabulary_build/1066/split'


    # 文件命名从001开始
    start_index = 1

    for color_img_path, binary_img_path in zip(color_images, binary_images):
        # 读取彩色图像和二值化图像
        color_img = cv2.imread(color_img_path)
        binary_img = cv2.imread(binary_img_path, cv2.IMREAD_GRAYSCALE)

        # 确保图像大小一致
        assert color_img.shape[:2] == binary_img.shape, "彩色图像和二值图像的尺寸不一致！"

        # 分割彩色图像和二值化图像
        color_patches = split_image(color_img, n, 1)
        binary_patches = split_image(binary_img, n, 2)

        # 保存彩色小图像
        save_patches(color_patches, original_folder, start_index)

        # 保存二值化小图像
        save_patches(binary_patches, split_folder, start_index)

        # 更新文件名索引
        start_index += len(color_patches)


if __name__ == '__main__':
    main()
