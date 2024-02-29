import os
import numpy as np
import math

os.system('cls')

def sigmoid(x):
    return (math.e**x)/(1+math.e**x)

x = np.array([  [1,1,1,0,1,1,0],
                [1,0,1,1,0,0,1],
                [1,0,0,0,1,1,0],
                [1,1,0,1,0,0,1],
                [1,1,1,0,1,0,1],
                [1,0,0,1,0,1,0],
                [1,0,1,0,1,0,1],
                [1,1,0,1,0,0,0],
                [1,0,1,0,1,1,1],
                [1,1,0,0,1,1,0]], dtype=np.float64)

y = np.array([0, 0, 1, 0, 1, 0, 1, 0, 0, 1], dtype=np.float64)

n = len(x[0])

weights = np.random.rand(n)

# จำนวน loop and learning_rat
loop = 5000 
learning_rat = 0.1

#loop ปรับค่า Weights 
for _ in range(loop):
    Predic = sigmoid((np.matmul(x, weights)))
    del_w = (2/n)*np.matmul((Predic-y)*Predic*(1-Predic),x)
    weights = weights - (learning_rat * del_w)

print(weights)