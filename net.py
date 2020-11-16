import math
import numpy as np
from graphics import*
import time

J = 0
dw = np.array([[0,0,0,0,0,0,0,0,0]])



db = 0
const = 0.000000000000000000000000000001
m = 19

x = np.array([[1, 1, 2, 7, 0, 1, 1, 9, 1, 0, 2, 0, 2, 1,  3,  0,  0,  5, 1],
              [6, 1, 1, 6, 2, 1, 0, 0, 3, 0, 6, 6, 0, 2,  1,  2,  1,  4, 6],
              [0, 0, 2, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0,  0,  1,  1,  1, 0],
              [1, 0, 1, 0, 0, 1, 1, 1, 2, 0, 0, 2, 0, 2,  0,  2,  0,  1, 4],
              [3, 2, 2, 0, 3, 1, 1, 0, 2, 0, 3, 4, 2, 2,  1,  1,  2,  3, 3],
              [0, 0, 1, 0, 1, 0, 1, 1, 0, 2, 0, 0, 0, 0,  0,  0,  0,  1, 1],
              [0, 1, 2, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 3,  1,  1,  1,  1, 2],
              [3, 0, 1, 0, 1, 1, 1, 3, 3, 4, 3, 0, 2, 1,  0,  1,  0,  1, 6],
              [0, 2, 2, 0, 2, 1, 1, 1, 2, 0, 0, 0, 0, 2,  1,  1,  0,  1, 2]])
              
y = np.array([[1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1,  0,  0,  0,  1, 1]])

z = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0, 0]])

a = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0, 0]])

dz= np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0, 0]])


w = np.array([[1,1,1,1,1,1,1,1,1]])
w = w.transpose()


b = 1

win = GraphWin('Vizualization', 700, 700)

msg = []
msg_eye = []

for i in range(9):
    yofc = (i+1)*60+50
    msg.append(Text(Point(180,yofc), '2'))
    msg_eye.append(Text(Point(50,yofc), '2'))
    obj = Circle(Point(180,yofc),25)
    eye = Circle(Point(50,yofc),25)
    line = Line(Point(75,yofc), Point(155, yofc))
    line2 = Line(Point(205,yofc), Point(285, 350))
    
    obj.setOutline('black')
    obj.setFill('red')
    eye.setOutline('black')
    eye.setFill(color_rgb(100, 100, 255))

    
    msg[i].setSize(10)
    msg[i].draw(win)
    obj.draw(win)
    eye.draw(win)
    msg_eye[i].draw(win)
    line.draw(win)
    line2.draw(win)


inner = Circle(Point(310,350), 25)
outlr = Circle(Point(440,350), 25)
inner.setOutline('black')
inner.setFill('yellow')
outlr.setOutline('black')
outlr.setFill(color_rgb(120, 255, 120))
msg_inner = Text(Point(310,350), '2')
msg_outlr = Text(Point(440,350), '2')
line3 = Line(Point(335,350), Point(415, 350))
inner.draw(win)
outlr.draw(win)
msg_inner.draw(win)
msg_outlr.draw(win)
line3.draw(win)
   
k=0
for j in range(2000):

    time.sleep(0)
    z = np.dot(w.T, x) + b
    print(z.shape)
    print('z = ', z)
    a = 1 / (1 + math.e**(-z))
    J = - np.sum(((y*np.log(a)) + ((1 - y)*np.log(1-a+const))))
        
    dz = a - y
        
    dw = np.dot(x, dz.T)

    db = np.sum(dz)

    J = J / m
    print('Overall loss for ', m,' training examples = ', J)
    
    dw = dw / m

    
    db = db / m
    
    i=0
    while len(msg)>0:
        msg[i].undraw()
        del msg[i]
        msg_eye[i].undraw()
        del msg_eye[i]
    msg_inner.undraw()

    for i in range(9):
        yofc = (i+1)*60+50
        tmp = '{:.3f}'.format(w[i,0])
        msg.append(Text(Point(180,yofc), tmp))
        msg_eye.append(Text(Point(50,yofc), x[i, k]))
        msg[i].setSize(10)
        msg[i].draw(win)
        msg_eye[i].draw(win)
    
    tmp2 = '{:.2f}'.format(z[0,k])
    msg_inner = Text(Point(310,350), tmp2)
    msg_inner.draw(win)
    msg_outlr.undraw()
    msg_outlr = Text(Point(440,350), '{:.2f}'.format(a[0,k]))
    msg_outlr.draw(win)
    k+=1
    if k == m:       k=0
    
    w = w - 0.1 * dw   
    b = b - 0.1 * db
 
fl = True
print('w: ', w)
print('b=', b)

print('Now let me guess!')
while fl:
    x_try = np.zeros(9)
    for i in range(9):
        print('x[', i+1, '] = ', end='')
        x_try[i] = float(input())

    
    f = np.dot(w.T, x_try) + b
    p = 1 / (1 + math.e**(-f))
    print('Probability = ', p)
    if p >= 0.9:
        print('Это наверняка единица')
    elif ((p > 0.6)and(p < 0.9)):
        print('Скорее всего единица')
    elif (p < 0.6)and(p > 0.4):
        print('Не знаю...')
    elif (p < 0.4)and(p > 0.1):
        print('Скорее всего это не единица')
    elif (p <= 0.1):
        print('Не, ну это точно не единица')
    print('Again? (y, n)')
    ch = input()
    if ch == 'n':
        fl = False
   

