import mahotas as mahotas

__author__ = 'altug'

import numpy as np
import argparse
import cv2

image = cv2.imread('test/testimg.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("Image", blurred)

thresh = cv2.adaptiveThreshold(blurred, 255,
cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
cv2.imshow("Mean Thresh", thresh)

thresh = cv2.adaptiveThreshold(blurred, 255,
cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
cv2.imshow("Gaussian Thresh", thresh)
'''
TERS TRESH
(T, thresh) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY)
(T, threshInv) = cv2.threshold(blurred, 155, 255, cv2.
THRESH_BINARY_INV)

cv2.imshow("Ters", threshInv)
cv2.imshow("Son Hali", cv2.bitwise_and(image, image, mask =
threshInv))
'''
'''
OTSU VE RIDDLE
T = mahotas.thresholding.otsu(image)
print "Otsu degeri: %d" % (T)

thresh = image.copy()
thresh[thresh > T] = 255
thresh[thresh < 255] = 0

thresh = cv2.bitwise_not(thresh)
cv2.imshow("Otsu", thresh)

T = mahotas.thresholding.rc(blurred)
print "Riddler-Calvard: %d" % (T)
thresh = image.copy()
thresh[thresh > T] = 255
thresh[thresh < 255] = 0
thresh = cv2.bitwise_not(thresh)
cv2.imshow("Riddler-Calvard", thresh)
'''
cv2.waitKey(0)
