import cv2
import csv

# 初始化一个列表来存储点击的点及其RGB值
points = []
current_label = 0

# 鼠标回调函数
def mouse_callback(event, x, y, flags, param):
    global current_label
    # 左键点击获取RGB值
    if event == cv2.EVENT_LBUTTONDOWN:
        rgb = img[y, x]
        points.append([rgb[2], rgb[1], rgb[0], current_label])
        print(f"Left Click: {rgb[2], rgb[1], rgb[0]}, Label: {current_label}")
    # 右键点击进入下一类
    elif event == cv2.EVENT_RBUTTONDOWN:
        current_label += 1
        print("进入下一类标注！")

# 读取图片
# 替换为你的图片路径
# img_path = '0618.png'
# img_path = '0854.png'
img_path = '1066.png'
img = cv2.imread(img_path)

# 创建一个窗口并设置鼠标回调
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)

while True:
    # 显示图片
    cv2.imshow('Image', img)
    # 等待键盘输入
    key = cv2.waitKey(1)
    # 按下Q键退出
    if key == ord('q'):
        break

# 打印收集到的点
print("Collected Points:")
for point in points:
    print(f"RGB: {point[0:3]}, Label: {point[3]}")

# 将收集到的点保存为CSV文件
# 设置CSV文件的路径
# csv_file_path = '0618.csv'
# csv_file_path = '0854.csv'
csv_file_path = '1066.csv'
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # 写入表头
    writer.writerow(['R', 'G', 'B', 'Label'])
    # 写入数据
    writer.writerows(points)

print(f"Points saved to {csv_file_path}")

# 释放资源
cv2.destroyAllWindows()
