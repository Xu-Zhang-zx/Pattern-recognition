import numpy as np
import cv2



def compare(gruth):
    sum_road = np.sum(gruth == 255)
    sum_not_road = np.sum(gruth == 0)
    pro_road = sum_road/(sum_road + sum_not_road)
    pro_not_road = sum_not_road/(sum_road + sum_not_road)
    print("Road:", pro_road)
    print("Not Road:", pro_not_road)
    return 1

if __name__ == "__main__":
    # 1. 加载文件

    # ture_img_path = '0618_truth.png'
    # ture_img_path = '0854_truth.png'
    ture_img_path = '1066_truth.png'


    truth_img = cv2.imread(ture_img_path, cv2.IMREAD_GRAYSCALE)

    compare(truth_img)
