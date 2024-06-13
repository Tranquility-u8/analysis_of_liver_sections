import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIBROTIC_MIN_AREA = 50000
IMAGE_HEIGHT = 1000
IMAGE_WIDTH = 1000


def getRatios(contour):
    rect = cv2.minAreaRect(contour)
    (_x, _y), (width, height), angle = rect
    return max(width, height) / min(width, height)


def getSolidity(contour):
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    contour_area = cv2.contourArea(contour)
    if hull_area == 0:
        return 0
    return float(contour_area) / hull_area


def getHuMoments(contour):
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments


def getContrast(contour, image):
    x, y, w, h = cv2.boundingRect(contour)
    region = image[y:y + h, x:x + w]

    # 计算图像的对比度
    glcm = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    glcm = cv2.bitwise_not(glcm)  # 反转图像，使背景为0，目标为255
    glcm = cv2.GaussianBlur(glcm, (5, 5), 0)  # 高斯模糊
    glcm = cv2.normalize(glcm, None, 0, 255, cv2.NORM_MINMAX)  # 归一化

    # 计算GLCM
    glcm_mat = cv2.calcHist([glcm], [0], None, [256], [0, 256])
    contrast = cv2.compareHist(glcm_mat, np.roll(glcm_mat, 1), cv2.HISTCMP_CORREL)

    return contrast


def getClass(hu_moments, area):
    if area < 100000:
        return "Normal"
    else:
        return "Severe Fibrosis"


def process_image(image_path, template_path, min_area=500):

    # Get Contour
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    template_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Init Table
    columns = ['Region', 'Area', 'Ratio', 'Solidity', 'Hu Moments', 'Match', 'Contrast', 'Class']
    results_df = pd.DataFrame(columns=columns)

    # Start Classification
    index = 1
    for i, contour in enumerate(contours):

        area = cv2.contourArea(contour)
        if area < FIBROTIC_MIN_AREA:
            continue

        # Shape Analysis
        hu_moments = getHuMoments(contour)
        ratio = getRatios(contour)
        solidity = getSolidity(contour)
        shape_class = getClass(hu_moments, area)
        match = cv2.matchShapes(contour, template_contours[1], cv2.CONTOURS_MATCH_I1, 0.0)

        # Texture Analysis
        contrast = getContrast(contour, image)

        M = cv2.moments(contour)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 10)
        cv2.putText(image, f"{index}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 8, (255, 0, 0), 15)
        results_df = pd.concat([
            results_df if not results_df.empty else None,
            pd.DataFrame([{
                'Region': index,
                'Area': area,
                'Ratio': ratio,
                'Solidity': solidity,
                'Hu Moments': hu_moments[0],
                'Match': match,
                'Contrast': contrast,
                'Class': shape_class,
            }])], ignore_index=True)
        index += 1

    # Display Result
    height, width = image.shape[:2]
    if height > IMAGE_HEIGHT or width > IMAGE_WIDTH:
        scaling_factor = IMAGE_HEIGHT / height if height > width else IMAGE_WIDTH / width
        image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(results_df)

    cv2.imshow("Shape Classification", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


process_image('test2.png', 'template.png', min_area=500)
