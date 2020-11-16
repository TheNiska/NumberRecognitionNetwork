import math
import numpy as np
from math import *
import time

m = 8 # 8 training examples

x = np.array([[2, 5, 1, 3, 4, 6, 1, 1],
              [7, 4, 1, 3, 4, 1, 6, 1],
              [2, 0, 1, 3, 4, 1, 1, 6]])


y = np.array([[1, 0, 0, 0, 1, 0, 0, 0]])

w1 = np.array([[0.32, -0.20, 0.14, 0.4],
               [0.10, 0.44, -0.31, 0.03],
               [0.15, -0.25, 0.41, -0.12]])


w1 = w1.T # now it is 4x3

b1 = np.array([[1],
               [-2],
               [3],
               [0.5]])

w2 = np.array([[0.2],
               [-0.3],
               [0.4],
               [-0.5]])
w2=w2.T
b2 = 1
                
for o in range(1500):

    print( '{:.2f}'.format(o/1500), '%')
    z1 = np.dot(w1, x) + b1 # size is 4xm = 4x8   
    a1 = (e**z1 - e**-z1) / (e**z1 + e**-z1)  # tanh function 4xm = 4x8
    print('a1=', a1)
    print('------------------------------------------------------')
    print('------------------------------------------------------')
    z2 = np.dot(w2, a1) + b2 # size is 1xm = 1x8
    a2 = 1 / (1 + e**(-z2))             # sigmoid function 1xm = 1x8
    print(a2)

    J = (1/m) * (- np.sum(((y*np.log(a2)) + ((1 - y)*np.log(1-a2)))))
    
    dz2 = a2 - y # shape is 1xm = 1x8
    dw2 = (1/m) * (np.dot(dz2, a1.T)) # shape is 1x4, i hope 4 is the number of nods in the hidden layer
    db2 = np.sum(dz2, axis=1, keepdims=True) * (1/m) # shape is 1x1 
    
    dz1 = np.dot(w2.T, dz2) * (1 - np.power(a1,2)) # shape is 4x8 = 4xm     
    dw1 = (1/m) * np.dot(dz1, x.T) # shape is 4x3 
    db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True) # shape is 4x1

    
    w2 = w2 - 0.1 * dw2 
    b2 = b2 - 0.1 * db2
    print('------------------------------------------------------')
    print('------------------------------------------------------')
    w1 = w1 - 0.1 * dw1
    b1 = b1 - 0.1 * db1
    
    
    print('w2=', w2)
    print('b2=', b2)

    print('w1=', w1)
    print('b1=', b1)
    print('------------------------------------------------------')
    print('------------------------------------------------------')
    print('Cost=', J)
    print('------------------------------------------------------')
    print('------------------------------------------------------')

print('Now let me guess!')
fl = True
while fl:
    x = np.array([[0],
                  [0],
                  [0]])
    for i in range(3):
        print('x[', i+1, '] = ', end='')
        x[i] = float(input())
    print(x)
    z1 = np.dot(w1, x) + b1
    a1 = (e**z1 - e**-z1) / (e**z1 + e**-z1)  # функция активации входного слоя - гиперболический арктангенс

    z2 = np.dot(w2, a1) + b2 
    a2 = 1 / (1 + e**(-z2))                   # функция активации выходного слоя - сигма
    
    print('p=', a2)
    
















