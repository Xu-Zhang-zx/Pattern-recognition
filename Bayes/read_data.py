import cv2
import csv

class ImageLabeler:
    def __init__(self, image_path, save_path):
        self.image_path = image_path
        self.save_path = save_path
        self.points = []
        self.image = cv2.imread(image_path)

        # 检查图片是否正确加载
        if self.image is None:
            raise ValueError("不能打开该图片!")

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self.mouse_callback)

        while True:
            cv2.imshow("Image", self.image)
            key = cv2.waitKey(1) & 0xFF
            # 按q键退出程序
            if key == ord('q'):
                self.save_to_csv()
                break

        cv2.destroyAllWindows()

    # 鼠标回调函数
    def mouse_callback(self, event, x, y, flags, param):
        # 左键点击,道路
        if event == cv2.EVENT_LBUTTONDOWN:
            # 获取RGB值
            rgb = self.image[y, x]
            # BGR转RGB，并添加标签0
            self.points.append((rgb[2], rgb[1], rgb[0], 0))
            print(f"左键点击: {rgb[2], rgb[1], rgb[0]} (Label: 0)")
            # 用红色标记
            cv2.circle(self.image, (x, y), 5, (255, 0, 0), -1)
        # 右键点击，非道路
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 获取RGB值
            rgb = self.image[y, x]
            # BGR转RGB，并添加标签1
            self.points.append((rgb[2], rgb[1], rgb[0], 1))
            print(f"右键点击: {rgb[2], rgb[1], rgb[0]} (Label: 1)")
            # 用蓝色标记
            cv2.circle(self.image, (x, y), 5, (0, 0, 255), -1)

    def save_to_csv(self):
        with open(self.save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # 写入表头
            writer.writerow(['R', 'G', 'B', 'Label'])
            # 写入数据
            writer.writerows(self.points)
        print("数据成功保存到" + self.save_path)

if __name__ == "__main__":
    # 路径
    # image_path = "0618.png"
    # save_path = "0618.csv"
    image_path = "0854.png"
    save_path = "0854.csv"
    # image_path = "1066.png"
    # save_path = "1066.csv"
    ImageLabeler(image_path, save_path)
