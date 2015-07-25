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
imagePaths = sorted(glob.glob("dataset/imagesjpg2" + "/*.jpg"))
maskPaths = sorted(glob.glob("dataset/masksjpg" + "/*.jpg"))

trainObj = TrainClassify(imagePaths, maskPaths)

#****************************************************TAC TEST***************************************************************
adaptiveObj = Adaptive()

imagePath = 'test/testimgm2.jpg'
maskedeneme = 'dataset/masks/mask_crocus_0001.png'
maskk = cv2.imread(maskedeneme)
image = cv2.imread(imagePath)
threshImage = adaptiveObj.getThresh(imagePath)
convertedThresh = cv2.cvtColor(threshImage, cv2.COLOR_GRAY2BGR)
rgbHistObj = RGBHistogram([8, 8, 8])
features = rgbHistObj.calculateHist(image, convertedThresh)

flower = trainObj.le.inverse_transform(trainObj.model.predict(features))[0] #Burada ilk feature i isim olarak aliyor bunu integerdan ceviriyor.
print np.shape(convertedThresh)
print imagePath

if flower == 'crocus': flower = 'cigdem'
if flower == 'daisy': flower = 'papatya'
if flower == 'pansy' : flower = 'menekse'
if flower == 'sunflower' : flower = 'aycicegi'

print "Bu cicek %s cinsi olabilir." % (flower.upper())

cv2.waitKey(0)