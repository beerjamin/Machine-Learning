import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

#Evaluation Procedure 1: Train and test on entire dataset
iris = load_iris()
X = iris.data
y = iris.target
#LogisticRegression
logreg = LogisticRegression()
logreg.fit(X,y)
logreg.predict(X)
y_pred = logreg.predict(X)
len(y_pred)
print 'LogisticRegression accuracy: ',metrics.accuracy_score(y, y_pred)

#Evaluation with KNN = 5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)
y_pred = knn.predict(X)
print 'Prediction where KNN = 5 accuracy: ', metrics.accuracy_score(y, y_pred)

#Evaluation with KNN = 1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X,y)
y_pred = knn.predict(X)
print 'Prediction where KNN = 1 accuracy: ',metrics.accuracy_score(y, y_pred)
