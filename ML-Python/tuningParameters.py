from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV

iris = load_iris()
X = iris.data
y = iris.target

knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
#print scores

#print scores.mean()
k_range = list(range(1,31))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
#for k in k_scores:
    #print k

param_grid = dict(n_neighbors=k_range)
#print param_grid

grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(X,y)
#for k in grid.grid_scores_:
#    print k

#print grid.grid_scores_[0].parameters
#print grid.grid_scores_[0].cv_validation_scores
#print grid.grid_scores_[0].mean_validation_score

grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
#print grid_mean_scores
k_range = list(range(1,31))
weight_options = ['uniform', 'distance']
param_grid = dict(n_neighbors=k_range, weights=weight_options)
#print param_grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(X,y)
#print grid.grid_scores_
print(grid.best_score_)
print(grid.best_params_)

knn = KNeighborsClassifier(n_neighbors=13, weights='uniform')
knn.fit(X,y)
print knn.predict([[3,5,4,2]])
print grid.predict([[3, 5, 4, 2]])

param_dist = dict(n_neighbors=k_range, weights=weight_options)
rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10, random_state=5)
rand.fit(X,y)
#print rand.grid_scores_
print(rand.best_score_)
print(rand.best_params_)

best_scores = []
for _ in range(20):
    rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10)
    rand.fit(X,y)
    best_scores.append(round(rand.best_score_,3))
print best_scores
