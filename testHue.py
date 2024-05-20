import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("test.jpg")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

h = hsv[:, :, 0].ravel()

plt.hist(h, 180, [0, 180])
plt.title("Histogram for hue")
plt.show()
