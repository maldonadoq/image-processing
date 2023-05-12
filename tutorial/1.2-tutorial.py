# Morphology

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('images/iris1.jpg', 0)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

dilate = cv2.dilate(img, kernel)

erode = cv2.erode(img, kernel)

image = cv2.absdiff(dilate, erode)

image = cv2.bitwise_not(image)

plt.imshow(image, cmap='gray')
plt.show()


img = np.array([[0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

kernel = np.array([[1., 1.],
                   [1., 1.]], dtype=np.uint8)

e = cv2.erode(img, kernel, iterations=1)
d = cv2.dilate(img, kernel, iterations=1)

e = cv2.erode(img, kernel, cv2.BORDER_REFLECT, iterations=1)


img = np.array([[0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0]], dtype=np.uint8)

kernel = np.array([[1, 0, 1]], dtype=np.uint8)


img = np.array([[0, 0, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 0, 1, 1, 0],
                [1, 1, 1, 1, 1, 0],
                [0, 1, 0, 1, 1, 1],
                [0, 1, 1, 1, 1, 0]], dtype=np.uint8)

kernel = np.array([[1, 0, 1]], dtype=np.uint8)

e = cv2.erode(img, kernel, anchor=(0, 0), iterations=1)
