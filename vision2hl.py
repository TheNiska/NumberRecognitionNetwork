from PIL import Image, ImageDraw
import numpy as np
from math import *
import matplotlib.pyplot as plt

m = 25 # i have 25 taining images 300x300 32 bit
ALFA = 0.001
LAMBD = 0    # regularization constant
EPSILON = 0.0000000000001
cost_func = []
x_iter = []


def image_to_vector(image: np.ndarray) -> np.ndarray:
    length, height, depth = image.shape
    return image.reshape((length * height * depth, 1))
 

for i in range(m):
    strr = 'images/train/' + str(i+1) + 't.png'          # images is named: 1t.png, 2t.png...
    
    img = Image.open(strr)
    try:
        data = np.asarray( img, dtype='uint8' )
    except SystemError:
        data = np.asarray( img.getdata(), dtype='uint8' )
    img.close()

    img_now = image_to_vector(data)
    n = 360000

    if i == 0:
        x = image_to_vector(data)
    else:
        x = np.column_stack((x, img_now))


y = np.array([[1,0,0,0,0,0,0,0,0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
#              1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25

# будем использовать нейронную сеть с двумя скрытыми слоями с количеством узлов 12 и 8
n0 = 360000
n1 = 10
n2 = 6
n3 = 1

np.random.seed(1)

a0 = x 
a0 = (a0 - 127.5) / 127.5

w1 = np.random.randn(n1, n0) * sqrt(1/n0)     # 12 x 360000
b1 = np.random.randn(n1,1)                    # 12 x 1

print(w1)

w2 = np.random.randn(n2,n1) * sqrt(1/n1)      # 8 x 12
b2 = np.random.randn(n2,1)                    # 8 x 1

w3 = np.random.randn(n3,n2) * sqrt(1/n2)      # 1 x 8
b3 = np.random.randn(n3,1)                    # 1 x 1


for o in range(2000):
    proc = (o / 2000) * 100
    print( '{:.2f}'.format(proc), '%')
    
    z1 = np.dot(w1, a0) + b1   # 12 x m
    a1 = np.tanh(z1)
    
    z2 = np.dot(w2, a1) + b2   # 8 x m
    a2 = np.tanh(z2)
    
    z3 = np.dot(w3, a2) + b3
    a3 = 1 / (1 + e**(-z3))    # 1 x m
    
    
    J = (1/m) * (- np.sum(((y*np.log(a3)) + ((1 - y)*np.log(1-a3)))))
    
    
    dz3 = a3 - y                                        # 1 x 25
    dw3 = (1/m) * (np.dot(dz3, a2.T)) + (LAMBD/m)*w3    # 1 x 8
    db3 = np.sum(dz3, axis=1, keepdims=True) * (1/m)    # 1 x 1
    
    dz2 = np.dot(w3.T, dz3) * (1 - np.power(a2, 2))     # 8 x 25
    dw2 = (1/m) * (np.dot(dz2, a1.T)) + (LAMBD/m)*w2    # 8 x 12
    db2 = np.sum(dz2, axis=1, keepdims=True) * (1/m)    # 8 x 1
    
    dz1 = np.dot(w2.T, dz2) * (1 - np.power(a1, 2))     # 12 x 25
    dw1 = (1/m) * (np.dot(dz1, a0.T)) + (LAMBD/m)*w1    # 12 x 360000
    db1 = np.sum(dz1, axis=1, keepdims=True) * (1/m)    # 12 x 1    


    w3 = w3 - ALFA * dw3 
    b3 = b3 - ALFA * db3
    
    w2 = w2 - ALFA * dw2 
    b2 = b2 - ALFA * db2

    w1 = w1 - ALFA * dw1
    b1 = b1 - ALFA * db1   
    
    
    print(J)
    Jtemp = '{:.4f}'.format(J)
    
    cost_func.append(float(Jtemp))
    x_iter.append(o)

    
    
fig = plt.subplots()  
plt.plot(x_iter, cost_func)
plt.show()

print('w3=', w3)
print('b3=', b3)  
 
print('w2=', w2)
print('b2=', b2)

print('w1=', w1)
print('b1=', b1)    


#-----VALIDATION---------------------------------------------------------------------------------
for i in range(10):
    strr = 'images/dev/t' + str(i+1) + '.png'    
    img = Image.open(strr)
    try:
        data = np.asarray( img, dtype='uint8' )
    except SystemError:
        data = np.asarray( img.getdata(), dtype='uint8' )
    img.close()

    img_now = image_to_vector(data)

    if i == 0: x = image_to_vector(data)
    else:      x = np.column_stack((x, img_now))

x = (x - 127.5) / 127.5
y = np.array([[0,0,0,0,0, 1,1,1,1,1]])

z1 = np.dot(w1, x) + b1
a1 = np.tanh(z1)

z2 = np.dot(w2, a1) + b2 
a2 = np.tanh(z2)
    
z3 = np.dot(w3, a2) + b3
a3 = 1 / (1 + e**(-z3))

J_dev = (1/10) * (- np.sum(((y*np.log(a3)) + ((1 - y)*np.log(1-a3)))))
print('Cost function in dev examples: ', J_dev)

#-------------------------------------------------------------------------------------------------

print('Now let me guess!')
fl = True
while fl:
    print('Enter name of a image:', end='')
    strr = input()
    strr ='images/' + strr
    img = Image.open(strr)
    
    try:
        data = np.asarray( img, dtype='uint8' )
    except SystemError:
        data = np.asarray( img.getdata(), dtype='uint8' )
    img.close()    
    
    img_now = image_to_vector(data)
    x = img_now
    x = x * 0.0001
    
    z1 = np.dot(w1, x) + b1
    a1 = np.tanh(z1)

    z2 = np.dot(w2, a1) + b2 
    a2 = np.tanh(z2)
    
    z3 = np.dot(w3, a2) + b3
    a3 = 1 / (1 + e**(-z3))
    
    print(a3)
    











