import numpy as np
import matplotlib.pyplot as plt
a = np.array([[1,1],[1.5,4]])
b = np.array([[2200],[5050]])
x = np.linalg.solve(a,b)
#print(x)

#MATPLOTLIB -> LIBRARY TO MAKE PLOTS VISUAL

x = np.linspace(0,10,20) #create datapoints
#0 is first argument, 10 second argument, 20 is number of points
#linspace - generate data that we havent seen yet

y = np.sin(x) #sin wave, take every element in x and apply sin to it

plt.plot(x,y)
#to show the plot, call the show function
#plt.show()
plt.plot(x,y)
plt.xlabel("Time")
plt.ylabel("Some function of time")
plt.title("My cool chart")
#plt.show()

x = np.linspace(0,10,100)
y = np.sin(x)
plt.plot(x,y)
#plt.show()
#if we use the same parameters, np.linspace() and arange() give the same results
