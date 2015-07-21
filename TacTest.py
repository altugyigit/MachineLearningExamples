__author__ = 'altug'

from RGBHistogram import RGBHistogram
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from TrainClassify import TrainClassify
import cv2
import glob

#****************************************************TAC EGITIM***************************************************************
imagePaths = sorted(glob.glob("dataset/imagesjpg" + "/*.jpg"))
maskPaths = sorted(glob.glob("dataset/masks" + "/*.png"))

trainObj = TrainClassify(imagePaths, maskPaths)

#****************************************************TAC TEST***************************************************************
imagePath = 'test/testimglale4.jpg'
image = cv2.imread(imagePath)

rgbHistObj = RGBHistogram([8, 8, 8])
features = rgbHistObj.calculateHist(image, mask=None)

flower = trainObj.le.inverse_transform(trainObj.model.predict(features))[0] #Burada ilk feature i isim olarak aliyor bunu integerdan ceviriyor.

print imagePath

if flower == 'crocus': flower = 'cigdem'
if flower == 'daisy': flower = 'papatya'
if flower == 'pansy' : flower = 'menekse'
if flower == 'sunflower' : flower = 'aycicegi'

print "Bu cicek %s cinsi olabilir." % (flower.upper())