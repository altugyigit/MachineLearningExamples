__author__ = 'altug'

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score

knn = KNeighborsClassifier(n_neighbors=5)

a = [1, 2, 3, 5, 6, 7]
b = [8, 9, 4]

scores = cross_val_score(knn, a, b, cv=10, scoring='accuracy')
print scores
