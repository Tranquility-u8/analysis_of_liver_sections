import cv2
import numpy as np


def calculate_hu_moments(contour):
    # 计算Hu矩
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments


def classify_shape(hu_moments, area):
    # 简单的形状分类逻辑
    # 你可以根据你的具体情况调整这些阈值
    if area < 500:
        return "Too Small"
    if hu_moments[0] < 0.002:
        return "Normal"
    elif hu_moments[0] < 0.005:
        return "Mild Fibrosis"
    else:
        return "Severe Fibrosis"


def process_image(image_path, min_area=500):
    # 读取图像并转换为灰度图像
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化图像
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        if area < min_area:
            continue  # 忽略面积小于最小面积的轮廓

        hu_moments = calculate_hu_moments(contour)
        shape_class = classify_shape(hu_moments, area)
        print(f"Region {i}:\nClass: {shape_class}\nArea: {area}\nHu Moments: {hu_moments}")

        # 绘制轮廓和形状分类结果
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # 标注序号
            cv2.putText(image, f"{i}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 显示结果图像
    cv2.imshow("Shape Classification", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 测试图像
process_image('test2.png', min_area=500)
