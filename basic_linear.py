#classwork 1 
import matplotlib.pyplot as plt
import numpy as np

def linear_regression(data_x, data_y):
    Sxx = (data_x - np.average(data_x)) ** 2
    Sxy = (data_x - np.average(data_x)) * (data_y - np.average(data_y))
    a = np.sum(Sxy) / np.sum(Sxx)
    b = np.average(data_y) - (np.average(data_x) * round(a, 2))
    return a, b

x = np.array([29, 28, 34, 31, 25])
y = np.array([77, 62, 93, 84, 59])
a, b = linear_regression(x, y)
regression = (a * x) + b

plt.scatter(x, y)
plt.plot(x, regression) 
plt.show()