__author__ = 'altug'

import numpy as np
import argparse
import cv2

image = cv2.imread('test/testimgh.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("Image", image)
cv2.waitKey()
