# Lines detection

import cv2
import numpy as np
import matplotlib.pyplot as plt

# read image
img = cv2.imread('images/watch.png')

if img.ndim == 3:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect edges
edges = cv2.Canny(gray, threshold1=100, threshold2=200, apertureSize=3)

# show edge map
plt.imshow(edges, cmap='gray')
plt.show()

hough_probabilistic = False
# hough_probabilistic = True

if hough_probabilistic:

    # apply probabilistic Hough transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi /
                            180, threshold=150, minLineLength=30, maxLineGap=10)

    # draw lines
    for i in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

else:
    # apply Hough transform
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=200)

    # draw lines
    for i in range(0, len(lines)):
        for rho, theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

# show results
plt.imshow(img)
plt.show()
# cv2.imwrite('lines_detected.jpg', img)
