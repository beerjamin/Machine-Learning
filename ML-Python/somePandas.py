import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

iris = load_iris()
type(iris)
X = iris.data
y = iris.target

logreg = LogisticRegression()
logreg.fit(X,y)
y_pred = logreg.predict(X)
#print 'LogisticRegression accuracy: ', metrics.accuracy_score(y,y_pred)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)
y_pred = knn.predict(X)
#print 'KNN = 5 prediction accuracy: ', metrics.accuracy_score(y,y_pred)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X,y)
y_pred = knn.predict(X)
#print 'KNN = 1 prediction accuracy: ', metrics.accuracy_score(y,y_pred)

X_test, X_train, y_test, y_train = train_test_split(X, y, test_size = 0.4)

scores = []
k_range = range(1,26)
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X,y)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
for p in scores:
    print p

plt.plot(k_range, scores)
plt.xlabel('Length of K')
plt.ylabel('Accuracy of predicition')
plt.show()
