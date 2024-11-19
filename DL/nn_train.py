from torch.utils.data import Dataset
import cv2
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision import transforms

# 定义的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Mydata(Dataset):
    def __init__(self, root_dir, img_dir, label_file, transform=None):
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.label_file = label_file
        self.transform = transform

        # 获取所有小图片的文件名，并按字母顺序排序
        self.img_path = sorted(os.listdir(os.path.join(self.root_dir, self.img_dir)))[:-1]

        # 读取标签文件，将标签保存到列表中
        with open(self.label_file, 'r') as f:
            self.labels = [int(line.strip()) for line in f.readlines()]

        # 确保标签数量与图片数量一致
        if len(self.img_path) != len(self.labels):
            raise ValueError("标签文件中的标签数量与图像数量不匹配!")

    def __getitem__(self, idx):
        # 获取图片文件名
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.img_dir, img_name)

        # 读取图片
        img = cv2.imread(img_item_path)
        if img is None:
            raise ValueError(f"无法读取图像：{img_item_path}")

        # 获取标签（根据文件顺序对应）
        label = self.labels[idx]

        # 转换为 RGB 并应用必要的预处理
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.img_path)



# 加载神经网络
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



# 数据集路径设置
train_root_dir = "dataset"  # 数据集的根目录
img_dir = "data_all"  # 小图片所在的文件夹（保存图像的地方）
label_file = "dataset/data_all/labels.txt"  # 标签文件（labels.txt）的路径

# 图像预处理（转换为 PyTorch 张量并做标准化）
transform = transforms.Compose([
    transforms.ToPILImage(),  # 转换为 PIL 图像
    transforms.Resize((128, 128)),  # 统一调整图像大小
    transforms.ToTensor(),  # 转换为 Tensor，并将像素值归一化到 [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 创建数据集对象
train_dataset = Mydata(train_root_dir, img_dir, label_file, transform=transform)
train_data_size = len(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=25, shuffle=True)

# # 获取数据集中的一张图片和其标签
# img, label = train_dataset[0]
# # 显示图片
# cv2.imshow('Img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # 打印标签
# print(label)

# 初始化模型
npic = NPIC()
npic = npic.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(npic.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 测试的轮数
epoch = 100
# 添加tensorboard
writer = SummaryWriter("logs_NPIC")

# 训练过程
for i in range(epoch):
    print("--------第{}轮训练开始--------".format(i + 1))
    npic.train()  # 设置模型为训练模式
    for imgs, targets in train_loader:
        # 将数据转移到 GPU
        imgs = imgs.to(device)
        targets = targets.to(device)

        # 前向传播
        outputs = npic(imgs)

        # 计算损失
        loss = loss_fn(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 优化步骤

        total_train_step += 1
        print(f"训练次数：{total_train_step}，loss：{loss.item()}")

        # 记录训练损失到 tensorboard
        writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    npic.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in train_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = npic(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集上的accuracy：{}".format(total_accuracy/train_data_size))
    writer.add_scalar("test_loss", loss.item(), total_test_step)
    writer.add_scalar("test_accurary", (total_accuracy/train_data_size).item(), total_test_step)
    total_test_step += 1

    # 每一轮训练结束后保存模型
    torch.save(npic.state_dict(), f"models/npic_{i}.pth")

writer.close()
