import cv2
from matplotlib import pyplot as plt


img_original = cv2.imread('test.jpg', 0)
# thresh = int(input())
thresh = 100
retval, img_global = cv2.threshold(img_original, thresh, 255, cv2.THRESH_BINARY)
print(retval)

ret2, th2 = cv2.threshold(img_original, 0, 255, cv2.THRESH_OTSU)
print(ret2)

plt.subplot(1, 3, 1)
plt.imshow(img_original, 'gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(th2, 'gray')
plt.title('Otsu thresholding')

plt.subplot(1, 3, 3)
plt.hist(img_original.ravel(), 256)

plt.show()
