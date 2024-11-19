import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import DictionaryLearning
from sklearn.linear_model import OrthogonalMatchingPursuit


def load_images_from_folder(folder, is_binary=False):
    """从文件夹中加载所有图像"""
    image_paths = glob.glob(os.path.join(folder, "*.png"))
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if is_binary:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).flatten()
            # 将像素值归一化到[0, 1]
            img = img / 255.0
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).flatten()
        images.append(img)
    return images


def split_image(image, patch_size):
    """将图像切分成小块"""
    h, w, c = image.shape
    patches = []
    for i in range(0, h, patch_size[0]):
        for j in range(0, w, patch_size[1]):
            patch = image[i:i + patch_size[0], j:j + patch_size[1]]
            if patch.shape[0] == patch_size[0] and patch.shape[1] == patch_size[1]:
                patches.append(patch.flatten())
    return patches

def visualize(original_image, reconstructed_image):
    """可视化原图和重建图"""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title('Reconstructed Image')
    plt.axis('off')

    plt.show()

# 输入图像和图库的文件夹路径
# color_image_folder = "dataset/image_vocabulary_build/0618/origin"
# binary_image_folder = "dataset/image_vocabulary_build/0618/split"
color_image_folder = "dataset/image_vocabulary_build/origin"
binary_image_folder = "dataset/image_vocabulary_build/split"

# input_image_path = "0618.png"
# input_image_path = "0854.png"
input_image_path = "1066.png"

# 加载词库
color_dictionary = load_images_from_folder(color_image_folder, is_binary = False)
binary_dictionary = load_images_from_folder(binary_image_folder, is_binary = True)

# 使用已有的字典
dict_learn = DictionaryLearning(transform_algorithm='omp', fit_algorithm='lars')
print(np.array(color_dictionary))
dict_learn.components_ = np.array(color_dictionary).T
dict_learn_binary = DictionaryLearning(transform_algorithm='omp', fit_algorithm='lars')
dict_learn_binary.components_ = np.array(binary_dictionary).T

# 读取输入大图
input_image = cv2.imread(input_image_path)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# 设置每个小图块的大小
# patch_size = (5, 5)
patch_size = (25, 25)

# 将大图切割成小块
color_patches = split_image(input_image, patch_size)

# 对每个图像块进行稀疏表示
omp = OrthogonalMatchingPursuit(n_nonzero_coefs = 500)
reconstructed_patches = []

for patch in color_patches:
    # 求解稀疏表示
    patch = np.array(patch).reshape(1, -1)
    omp.fit(dict_learn.components_, patch.T)
    # 获取稀疏系数
    sparse_code = omp.coef_
    # print(sparse_code)
    # 用稀疏系数重建图像块
    reconstructed_patch = np.dot(sparse_code, dict_learn_binary.components_.T)
    reconstructed_patch = (reconstructed_patch > 0.5).astype(int)

    reconstructed_patches.append(reconstructed_patch)

# 重建完整的图像（将图像块拼接）
h, w, _ = input_image.shape
reconstructed_image = np.zeros((h, w), dtype=np.uint8)
idx = 0
for i in range(0, h, patch_size[0]):
    for j in range(0, w, patch_size[1]):
        if idx < len(reconstructed_patches):
            patch = reconstructed_patches[idx].reshape(patch_size[0], patch_size[1])
            reconstructed_image[i:i + patch_size[0], j:j + patch_size[1]] = patch*255
            idx += 1

# 可视化原图和重建图像
visualize(input_image, reconstructed_image)
