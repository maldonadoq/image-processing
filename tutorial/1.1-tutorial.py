# circle detection

import cv2
import numpy as np
import matplotlib.pyplot as plt

# read image
img = cv2.imread('images/coins.png')
output = img.copy()

if img.ndim == 3:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect edges
edges = cv2.Canny(gray, threshold1=100, threshold2=200, apertureSize=3)

# show edge map
plt.imshow(edges, cmap='gray')
plt.show()

# apply Hough transform
circles = cv2.HoughCircles(gray, method=cv2.HOUGH_GRADIENT, dp=1, minDist=40,
                           param1=50, param2=30, minRadius=10, maxRadius=40)

if circles is not None:

    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")

    # extract (x, y) coordinates and radius from the circles
    for (x, y, r) in circles:
        # draw the circle and its corresponding center
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.rectangle(output, (x-2, y-2), (x+2, y+2), (0, 128, 255), -1)

# show results
plt.imshow(output)
plt.show()
# cv2.imwrite('circles_detected.jpg', img)
