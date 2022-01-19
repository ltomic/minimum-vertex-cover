import sys
import os
import random

for i in os.listdir('weighted'):
    f = open("weighted/"+i, 'r+')
    s = f.readline()
    v = int(s.split(' ')[2])
    f.seek(0, 2)
    for i in range(v):
        f.write("v {} {}\n".format(i+1, random.randint(20, 100)))
    f.close()
