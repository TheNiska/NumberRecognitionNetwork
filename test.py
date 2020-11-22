import numpy as np

def file_data_clear(filename: str):
    file = open(filename, 'r')
    string_array = ''
    line = file.readline()
    while line:
        line = line.replace(']', '')
        line = line.replace('[', '')
        line = line.strip(' ')
        string_array += line
        line = file.readline()
    file.close()
    file = open(filename, 'w')
    file.write(string_array)
    file.close()

'''
file = open('weights.txt', 'w')
a = np.random.randn(10, 5)

file.write(str(a))
file.close() 
   
file_data_clear('weights.txt')

a = np.loadtxt("weights.txt")
print(a)
print(a.shape)
'''


