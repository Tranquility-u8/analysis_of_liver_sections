import cv2
import numpy as np

original_image = cv2.imread('test.png')
segmented_image = cv2.imread('filter.png', 0)

contours, _ = cv2.findContours(segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contoured_image = original_image.copy()
cv2.drawContours(contoured_image, contours, -1, (0, 255, 0), 3)

cv2.imshow('Original Image', original_image)
cv2.imshow('Segmented Image', segmented_image)
cv2.imshow('Contoured Image', contoured_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
