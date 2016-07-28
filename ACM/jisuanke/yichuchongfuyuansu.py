# -*- coding: utf-8 -*-
#移除数组中的重复元素 
num = raw_input()

line = raw_input()
line = line.split()
count = 0
for i in xrange(1,len(line)):
    if(line[i-1] != line[i]):
        count = count + 1
print count+1