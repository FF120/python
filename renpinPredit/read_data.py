# -*- coding: utf-8 -*-
import sys

list = []
dict = {}
import csv
import numpy as np


csvfile = file('data/train_x.csv', 'rb')
reader = csv.reader(csvfile)
list = []
for line in reader:
    if reader.line_num == 1:  
        continue  
    line = map(eval, line)
    list.append(line)
tran_x = np.array(list)
np.save("data/train_x.npy",tran_x)
csvfile.close() 

csvfile2 = file('data/train_y.csv','rb')
reader2 = csv.reader(csvfile2)
list2 = []
for line2 in reader2:
    if reader2.line_num == 1:
        continue
    line2 = map(eval,line2)
    list2.append(line2)
tran_y = np.array(list2)
np.save("data/train_y.npy",tran_y)
csvfile2.close()

