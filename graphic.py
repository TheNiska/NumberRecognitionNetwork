
import numpy as np
import matplotlib.pyplot as plt

y = lambda x: np.sin(x)

fig = plt.subplots()

x = np.linspace(0, 1000,100)
# значения x, которые будут отображены
# количество элементов в созданном массиве
# - качество прорисовки графика 
# рисуем график
plt.plot(([1,2,3,4,5]), ([1,4,9,16,25]))
# показываем график
plt.show()