__author__ = 'altug'

from RGBHistogram import RGBHistogram
from Adaptive import Adaptive
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from TrainClassify import TrainClassify
import cv2
import glob
import numpy as np

#****************************************************TAC EGITIM***************************************************************
imagePaths = sorted(glob.glob("dataset/imagesjpg" + "/*.jpg"))#Egitim path i.
maskPaths = sorted(glob.glob("dataset/masksjpg" + "/*.jpg"))

trainObj = TrainClassify(imagePaths, maskPaths)

#****************************************************TAC TEST***************************************************************
#adaptiveObj = Adaptive()
testPaths = sorted(glob.glob("test" + "/*.jpg"))

for testPath in testPaths:
    image = cv2.imread(testPath)
    #threshImage = adaptiveObj.getThresh(imagePath)
    #convertedThresh = cv2.cvtColor(threshImage, cv2.COLOR_GRAY2BGR)
    #masked_img = cv2.bitwise_and(image,image,mask = threshImage)
    #cv2.imshow("asdasd", masked_img)
    rgbHistObj = RGBHistogram([8, 8, 8])
    features = rgbHistObj.calculateHist(image, mask=None)

    flower = trainObj.le.inverse_transform(trainObj.model.predict(features))[0] #Burada ilk feature i isim olarak aliyor bunu integerdan ceviriyor.

    print testPath

    if flower == 'crocus': flower = 'cigdem'
    if flower == 'daisy': flower = 'papatya'
    if flower == 'pansy' : flower = 'menekse'
    if flower == 'sunflower' : flower = 'aycicegi'

    print "Bu cicek %s cinsi olabilir." % (flower.upper())