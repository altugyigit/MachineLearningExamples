__author__ = 'altug'

from RGBHistogram import RGBHistogram
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import argparse
import glob
import cv2

'''ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required = True,
help = "path to the image dataset")
ap.add_argument("-m", "--masks", required = True,
help = "path to the image masks")
args = vars(ap.parse_args())'''

imagePaths = sorted(glob.glob("dataset/images" + "/*.png"))
maskPaths = sorted(glob.glob("dataset/masks" + "/*.png"))

data = []
target = []

rgbHistObj = RGBHistogram([8, 8, 8])

for (imagePath, maskPath) in zip(imagePaths, maskPaths):
    image = cv2.imread(imagePath)
    mask = cv2.imread(maskPath)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    features = rgbHistObj.calculateHist(image, mask = None)#Maskta size hatasi veiyor none yaptim.

    data.append(features)
    target.append(imagePath.split("_")[1])

#Machine Learning
targetNames = np.unique(target)#Tekrarlari at.
le = LabelEncoder()
target = le.fit_transform(target)#Tur isimlerini integer a cevir mach. lear. icin

#Random Train ve Test datasi olustur elimizdeki verecegimiz egitim setinin overfittting olmamasi icin.
#Datanin dogrulugunu kontrol icin AccuracyTest kodu yazildi.
(trainData, testData, trainTarget, testTarget) = train_test_split(data, target, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=25, random_state=84)
model.fit(trainData, trainTarget) # Egitim gerceklesiyor.

print '*********************************************EGITIM VERLERI************************************************\n' +\
      classification_report(testTarget, model.predict(testData), target_names = targetNames)+\
    '************************************************************************************************************'

#BURAYA BIZIM CEKILEN FOTOGRAF GELECEK ADRIAN AYNI DATASET ILE KARSILASTIRMIS.
'''for i in np.random.choice(np.arange(0, len(imagePaths)), 10):
        imagePath = imagePaths[i]
        maskPath = maskPaths[i]

        image = cv2.imread(imagePath)
        mask = cv2.imread(maskPath)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)'''

imagePath = 'testimg.jpg'
image = cv2.imread(imagePath)

features = rgbHistObj.calculateHist(image, mask=None)

flower = le.inverse_transform(model.predict(features))[0] #Burada ilk feature i isim olarak aliyor bunu integerdan ceviriyor.
print imagePath
print "I think this flower is a %s" % (flower.upper())







