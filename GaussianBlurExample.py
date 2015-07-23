import mahotas as mahotas

__author__ = 'altug'

import numpy as np
import argparse
import cv2

image = cv2.imread('test/testimg.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("Image", blurred)

T = mahotas.thresholding.otsu(blurred)
print "Otsu degeri: %d" % (T)

thresh = image.copy()
thresh[thresh > T] = 255
thresh[thresh < 255] = 0

thresh = cv2.bitwise_not(thresh)
cv2.imshow("Otsu", thresh)


cv2.waitKey()
