import numpy as np
import test as tst

for i in range(10):
    file_name = 'subm/num' + str(i) + '.txt'
    tst.file_data_clear(file_name)
    print('|',end='')

max = np.zeros((28000,1))
imax = np.zeros((28000,1),dtype=int)

for i in range(10):
    file_name = 'subm/num' + str(i) + '.txt'
    numbers = np.loadtxt(file_name)
    numbers = numbers.reshape(28000,1)
    
    for j in range(28000):
        if numbers[j,0] > max[j,0]:
            max[j,0] = numbers[j,0]
            imax[j,0] = i
            
fil = open('submiss.csv','w')
fil.write('ImageId,Label\n')
for i in range(28000):
    line = str(i+1) + ',' + str(imax[i,0])+ '\n'
    fil.write(line)



