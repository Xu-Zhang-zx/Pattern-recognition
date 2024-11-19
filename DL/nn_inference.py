import cv2
import numpy as np
import torch
from torchvision import transforms
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NPIC(nn.Module):
    def __init__(self):
        super(NPIC, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=2),  # 输入通道3，输出32，卷积核大小3，padding=2
            nn.MaxPool2d(2),  # 最大池化，尺寸减半
            nn.Conv2d(32, 32, 5, padding=2),  # 输入通道32，输出32
            nn.MaxPool2d(2),  # 最大池化，尺寸减半
            nn.Conv2d(32, 64, 5, padding=2),  # 输入通道32，输出64
            nn.MaxPool2d(2),  # 最大池化，尺寸减半
            nn.Flatten(),  # 将多维数据展平为一维
            nn.Linear(64 * 16 * 16, 64),  # 经过卷积和池化后，图像尺寸为16x16，通道数为64
            nn.Linear(64, 2)  # 输出2个类别
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 加载训练好的模型
model_path = 'models/npic_61.pth'  # 替换为实际的模型路径
npic = NPIC()
npic.load_state_dict(torch.load(model_path))
npic = npic.to('cuda' if torch.cuda.is_available() else 'cpu')  # 使用 GPU 或 CPU

# 图像预处理（转换为 PyTorch 张量并做标准化）
transform = transforms.Compose([
    transforms.ToPILImage(),  # 转换为 PIL 图像
    transforms.Resize((128, 128)),  # 统一调整图像大小
    transforms.ToTensor(),  # 转换为 Tensor，并将像素值归一化到 [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# # 推理并修改图片中心像素值的函数
# def predict_and_modify_image(image_path, model, n, transform=None):
#     """
#     对一张图片进行推理，并将每个n×n小图片的中心像素值修改为0或255
#     :param image_path: 输入图片路径
#     :param model: 已训练好的模型
#     :param n: 小图片的尺寸
#     :param transform: 图像预处理方法
#     :return: 修改后的图像
#     """
#     # 读取图像
#     img = cv2.imread(image_path)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#     # 获取图像的高度和宽度
#     height, width, _ = img_rgb.shape
#
#     # 新图像初始化为原始图像的副本
#     modified_img = img.copy()
#
#     # 滑动窗口大小为1
#     half_n = n // 2
#     for y in range(half_n, height - half_n, 1):  # y轴滑动
#         for x in range(half_n, width - half_n, 1):  # x轴滑动
#             # 提取 n×n 的小图片
#             small_img = img_rgb[y - half_n:y + half_n + 1, x - half_n:x + half_n + 1]
#
#             # 如果提供了transform，则应用
#             if transform:
#                 small_img_transformed = transform(small_img).unsqueeze(0).to(device)
#             else:
#                 # 直接转换为 Tensor
#                 small_img_transformed = transforms.ToTensor()(small_img).unsqueeze(0).to(device)
#
#             # 推理
#             with torch.no_grad():
#                 output = model(small_img_transformed)
#                 _, predicted = torch.max(output, 1)
#
#             # 根据预测的标签修改中心像素的值
#             if predicted.item() == 0:
#                 # 将中心点像素值设为 0
#                 modified_img[y, x] = [0, 0, 0]
#             else:
#                 # 将中心点像素值设为 255
#                 modified_img[y, x] = [255, 255, 255]
#     return modified_img

def predict_and_modify_image(image_path, model, n, transform=None):
    """
    对一张图片进行推理，并将每个n×n小图片的中心像素值修改为0或255
    :param image_path: 输入图片路径
    :param model: 已训练好的模型
    :param n: 小图片的尺寸
    :param transform: 图像预处理方法
    :param device: 设备（'cpu' 或 'cuda'）
    :return: 修改后的图像
    """
    # 读取图像
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 获取图像的高度和宽度
    height, width, _ = img_rgb.shape

    # 新图像初始化为原始图像的副本
    modified_img = img.copy()

    # 计算滑动窗口的半宽度
    half_n = n // 2

    # 对于图像的每个小区域进行处理
    for y in range(0, height):  # 遍历 y 轴
        for x in range(0, width):  # 遍历 x 轴
            # 计算当前图像块的范围，确保在边缘部分也能有效处理
            y_start = max(0, y - half_n)
            y_end = min(height, y + half_n + 1)
            x_start = max(0, x - half_n)
            x_end = min(width, x + half_n + 1)

            # 提取 n×n 的小图片块
            small_img = img_rgb[y_start:y_end, x_start:x_end]

            # 如果提供了transform，则应用
            if transform:
                small_img_transformed = transform(small_img).unsqueeze(0).to(device)
            else:
                # 直接转换为 Tensor
                small_img_transformed = transforms.ToTensor()(small_img).unsqueeze(0).to(device)

            # 推理
            with torch.no_grad():
                output = model(small_img_transformed)
                _, predicted = torch.max(output, 1)

            # 根据预测的标签修改中心像素的值
            if predicted.item() == 0:
                # 将中心点像素值设为 0
                modified_img[y, x] = [0, 0, 0]
            else:
                # 将中心点像素值设为 255
                modified_img[y, x] = [255, 255, 255]

    return modified_img

# 主函数：加载图像，推理并展示原图与推理后的图像
def main(image_path):
    # 设置小图片尺寸 n
    n = 25

    # 调用推理函数，获取修改后的图像
    modified_img = predict_and_modify_image(image_path, npic, n, transform=transform)

    # 读取原始图像，用于展示
    original_img = cv2.imread(image_path)

    # 将原图和推理后图像转换为 RGB
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    modified_img_rgb = cv2.cvtColor(modified_img, cv2.COLOR_BGR2RGB)

    # 展示原图与推理后的图像
    combined = np.hstack((original_img_rgb, modified_img_rgb))  # 水平拼接原图和修改后的图像
    cv2.imshow("Original and Modified Image", combined)

    # 等待按键事件
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 调用主函数，输入图像路径
if __name__ == '__main__':
    # image_path = "0618.png"
    # image_path = "0854.png"
    image_path = "1066.png"
    main(image_path)
