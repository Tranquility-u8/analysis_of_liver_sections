from tkinter import filedialog

import cv2
import numpy as np
import pandas as pd
from skimage import feature
import matplotlib.pyplot as plt

REGION_MIN_AREA = 40000
FIBROTIC_MIN_AREA = 100000
FIBROTIC_HOMOGENEITY = 0.5
IMAGE_HEIGHT = 1000
IMAGE_WIDTH = 1000

region_class = ["正常", "纤维化(一级)", "纤维化(二级)", "纤维化(三级)"]


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


def getGLCMFeature(gray_image, contour):
    x, y, w, h = cv2.boundingRect(contour)
    gray_region = gray_image[y:y + h, x:x + w]

    glcm = feature.graycomatrix(gray_region, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], 256, symmetric=True, normed=True)
    contrast = feature.graycoprops(glcm, 'contrast')
    homogeneity = feature.graycoprops(glcm, 'homogeneity')
    energy = feature.graycoprops(glcm, 'energy')
    return homogeneity


def getCavities(gray_image):
    # x, y, w, h = cv2.boundingRect(contour)
    # gray_region = gray_image[y:y + h, x:x + w]

    inverted_img = cv2.bitwise_not(gray_image)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(inverted_img, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(inverted_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def getRegionClass(area, ratio, solidity, hu_moments, homogeneity):
    if area < FIBROTIC_MIN_AREA:
        return region_class[0]
    if homogeneity < 0.6:
        return region_class[0]

    if ratio > 3.5:
        return region_class[3]
    if ratio > 2.8 and solidity < 0.7:
        return region_class[3]

    if area > 300000 and solidity + hu_moments[0] < 0.9:
        return region_class[2]
    # if area > 300000 and solidity < 0.6 and hu_moments[0] < 0.35:
    #     return region_class[2]

    return region_class[1]


def process_image(image_path, template_path):
    # Get Contour
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    # template_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Init Table
    columns = ['Region', 'Area', 'Ratio', 'Solidity', 'Hu Moments', 'homogeneity', 'Class']
    results_df = pd.DataFrame(columns=columns)

    # Start Classification
    index = 1
    for i, contour in enumerate(contours):

        area = cv2.contourArea(contour)
        if area < REGION_MIN_AREA:
            continue

        # Shape Analysis
        ratio = getRatios(contour)
        solidity = getSolidity(contour)
        hu_moments = getHuMoments(contour)
        # match = cv2.matchShapes(contour, template_contours[1], cv2.CONTOURS_MATCH_I1, 0.0)

        # Texture Analysis
        homogeneity = getGLCMFeature(gray_image, contour)
        shape_class = getRegionClass(area, ratio, solidity, hu_moments, homogeneity)

        # Draw Contour
        M = cv2.moments(contour)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 10)
        cv2.putText(image, f"{index}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 8, (255, 0, 0), 15)

        # Display Result
        results_df = pd.concat([
            results_df if not results_df.empty else None,
            pd.DataFrame([{
                'Region': index,
                'Area': area,
                'Ratio': ratio,
                'Solidity': solidity,
                'Hu Moments': hu_moments[0],
                'Homogeneity': homogeneity[0][0],
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


file_path = filedialog.askopenfilename()
process_image(file_path, 'template.png')
