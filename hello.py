import numpy as np

a = open('/home/seungkwan/ubbr_data.txt', 'r')
for line in a.readlines():
    print(np.sum(list(map(float, line.split()))) / 20)

