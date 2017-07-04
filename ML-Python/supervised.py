import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

#First method of KNN = 1
iris = load_iris()
type(iris)
X = iris.data #matrix
y = iris.target #vector
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X,y)
X_new = [[3,5,4,2],[5,4,3,2]]
print 'Prediction where KNN = 1: ',knn.predict(X_new)

#Second method where KNN = 5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)
print 'Prediction where KNN = 5: ',knn.predict(X_new)

#Third method using LogisticRegression
logreg = LogisticRegression()
logreg.fit(X,y)
print 'LogisticRegression: ',logreg.predict(X_new)
