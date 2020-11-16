from PIL import Image, ImageDraw
import numpy as np
from math import *
import matplotlib.pyplot as plt
m = 25 # i have 12 taining images 300x300 32 bit
cost_func = []
x_iter = []

for i in range(m):
    strr = 'images/' + str(i+1) + 't.png'          # images is named: 1t.png, 2t.png...
    img = Image.open(strr)
    #------------------------------------------------------------------
    def image_to_vector(image: np.ndarray) -> np.ndarray:
        """
        Args:
        image: numpy array of shape (length, height, depth)        # this part of code I found in the internet

        Returns:
         v: a vector of shape (length x height x depth, 1)
        """
        length, height, depth = image.shape
        return image.reshape((length * height * depth, 1))
    try:
        data = np.asarray( img, dtype='uint8' )
    except SystemError:
        data = np.asarray( img.getdata(), dtype='uint8' )
    img.close()
    
    #-------------------------------------------------------------------

    img_now = image_to_vector(data)
    n = 360000

    if i == 0: x = image_to_vector(data)
    else:      x = np.column_stack((x, img_now)) # a way to stack together all images into one X matrix
    print(x)
    print(x.shape)

y = np.array([[1,0,0,0,0,0,0,0,0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
#              1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25
# (first image is 1, second is 2....., 10-12 is 1)

w1 = np.random.randn(360000, 20) * 0.01 # random inizsidfiasdfassialzialization
w1 = w1.T

b1 = np.random.randn(20,1)

w2 = np.random.randn(20,1) * 0.01
w2 = w2.T

b2 = 0.5
x = x * 0.00001

for o in range(3000):
    proc = (o / 3000) * 100
    print( '{:.2f}'.format(proc), '%')
    z1 = np.dot(w1, x) + b1
    # a1 = (e**z1 - e**-z1) / (e**z1 + e**-z1)  # tanh function 4xm = 4x8
    a1 = np.tanh(z1)
   
    z2 = np.dot(w2, a1) + b2
    a2 = 1 / (1 + e**(-z2)) 
    print('a2=', a2)
    print('------------------------------------------------------')
    J = (1/m) * (- np.sum(((y*np.log(a2)) + ((1 - y)*np.log(1-a2)))))
    
    dz2 = a2 - y
    dw2 = (1/m) * (np.dot(dz2, a1.T)) 
    db2 = np.sum(dz2, axis=1, keepdims=True) * (1/m) 

    dz1 = np.dot(w2.T, dz2) * (1 - np.power(a1,2))    
    dw1 = (1/m) * np.dot(dz1, x.T) 
    db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)

    
    w2 = w2 - 0.1 * dw2 
    b2 = b2 - 0.1 * db2

    w1 = w1 - 0.1 * dw1
    b1 = b1 - 0.1 * db1
    print('Cost===========================================', J)
    
    Jtemp = '{:.4f}'.format(J)  
    cost_func.append(float(Jtemp))
    x_iter.append(o)
    
    print('------------------------------------------------------')
    print('------------------------------------------------------')    
'''  
    print('w2=', w2)
    print('b2=', b2)

    print('w1=', w1)
    print('b1=', b1)
    print('Cost===========================================', J)
    print('------------------------------------------------------')
    print('------------------------------------------------------')
'''
print('w2=', w2)
print('b2=', b2)

print('w1=', w1)
print('b1=', b1)

fig = plt.subplots()  
plt.plot(x_iter, cost_func)
plt.show()

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
    x = x * 0.00001
    
    z1 = np.dot(w1, x) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = 1 / (1 + e**(-z2))
    print(a2)
    
    
    
    
    
    
    
    
    
    