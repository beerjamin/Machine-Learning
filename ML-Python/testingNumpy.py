import numpy as np

#print(a)

def printer(a):
    for e in a:
        print(e)
#printer(a)
a = np.array([1,2])
b = np.array([2,1])

def dot(a,b):
    res = 1
    for e,f in zip(a,b):
        res *= e*f
    return res
#print(dot(a,b))
#print(np.sum(a*b))
#print((a*b).sum())
#print(np.dot(a,b))
#print(a.dot(b))
amag = np.sqrt((a*a).sum())
#print(amag)
amag = np.linalg.norm(a)
#print(amag)
cosangle = a.dot(b) / (np.linalg.norm(a)*np.linalg.norm(b))
#print(cosangle)
angle = np.arccos(cosangle)
#print(angle)
