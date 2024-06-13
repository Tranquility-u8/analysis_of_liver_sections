import cv2


image = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('people', image)
cv2.waitKey(0)

people_down_1 = cv2.pyrDown(image)
cv2.imshow('people_down_1', people_down_1)

people_down_2 = cv2.pyrDown(people_down_1)
cv2.imshow('people_down_2', people_down_2)

cv2.waitKey(0)
