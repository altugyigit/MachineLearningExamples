__author__ = 'altug'

import glob,cv2

imagePaths = sorted(glob.glob("dataset/gul" + "/*.jpg"))#RENAME OLACAK PATH.

i=1
for imageName in imagePaths:
    image = cv2.imread(imageName)
    cv2.imwrite("dataset/imagesjpg2/train_gul_"+str(i)+".jpg", image)
    i=i+1

