import numpy as np

m = np.array([[1,2], [3,4]])
#CONVENTION --> first index is row, second index is column
l = [[1,2],[3,4]]
print l[0][0]
print m[0][0]
print m[0,0]
m2 = np.matrix([[1,2],[3,4]]) #it is more conventional to use array instead of matrix
#print(m2)
a = np.array(m2)
#print(a)
#print(a.T) --> transpose

#CREATE ARRAYS
#array of zeros example
z = np.zeros(10)
#print(z)
z = np.zeros((10,10))
#print(z)
o = np.ones((10,10))
#print(o)
r = np.random.random((10,10)) #0 < x < 1
#probability distrubution to know where the numbers came from
#print(r)

#gaussian distributed numbers
g = np.random.randn(10,10)
#print(g)
g.mean() #find mean
g.var() #find variance

#MATRIX OPERATIONS
A = np.array([[1,2],[3,4]])
Ainv = np.linalg.inv(A) #matrix inverse
#print(Ainv)
##print(A.dot(Ainv))
np.linalg.det(A)
#print(np.linalg.det(A)) determinant

np.diag(A) #diagonal elements
#print (np.diag([1,2]))

a = np.array([1,2])
b = np.array([3,4])
np.outer(a,b) #outer product
np.inner(a,b) #inner product or a.dot(b)
np.diag(A).sum() #trace of A
np.trace(A)

X = np.random.randn(100,3) #each sample takes up a row, each column is a feature
#100 samples, 3 features
cov = np.cov(X)
#print cov.shape --> this is wrong
cov = np.cov(X.T) #covariants of a data matrix
#print(cov)

#print np.linalg.eigh(cov)
#print np.linalg.eig(cov)

#SOLVING A LINEAR EQUATION
b = np.array([1,2])
x = np.linalg.inv(A).dot(b)#dont use inverse because the solve() method is more efficient
#print x
x = np.linalg.solve(A,b)
print(x)
