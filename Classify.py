import cv2
import numpy as np

MIN_AREA = 50000


def calculate_hu_moments(contour):
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments


def classify_shape(hu_moments, area):
    if area < 100000:
        return "Normal"
    else:
        return "Severe Fibrosis"


def calculate_contrast(image):
    # 计算图像的对比度
    glcm = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = cv2.bitwise_not(glcm)  # 反转图像，使背景为0，目标为255
    glcm = cv2.GaussianBlur(glcm, (5, 5), 0)  # 高斯模糊
    glcm = cv2.normalize(glcm, None, 0, 255, cv2.NORM_MINMAX)  # 归一化

    # 计算GLCM
    glcm_mat = cv2.calcHist([glcm], [0], None, [256], [0, 256])
    contrast = cv2.compareHist(glcm_mat, np.roll(glcm_mat, 1), cv2.HISTCMP_CORREL)

    return contrast


def process_image(image_path, template_path, min_area=500):
    # 读取图像和模板图，并转换为灰度图像
    image = cv2.imread(image_path)

    # 读取图像并转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化图像
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 查找模板图的轮廓
    template_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 查找图像的轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    index = 1
    for i, contour in enumerate(contours):
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue  # 忽略面积小于最小面积的轮廓

        match = cv2.matchShapes(contour, template_contours[1], cv2.CONTOURS_MATCH_I1, 0.0)

        hu_moments = calculate_hu_moments(contour)
        shape_class = classify_shape(hu_moments, area)
        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y:y+h, x:x+w]  # 获取轮廓包围的区域
        contrast = calculate_contrast(roi)

        M = cv2.moments(contour)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 10)
        cv2.putText(image, f"{index}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 8, (255, 0, 0), 15)
        print("=======================================================")
        print(f"Region {index}:\nClass: {shape_class}\nArea: {area}\nHu Moments: {hu_moments}\nMatch:{match}\nContrast:{contrast}\n")
        index += 1

    height, width = image.shape[:2]
    max_height = 1000
    max_width = 1000
    if height > max_height or width > max_width:
        scaling_factor = max_height / height if height > width else max_width / width
        image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    cv2.imshow("Shape Classification", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


process_image('test2.png', 'template.png', min_area=500)

