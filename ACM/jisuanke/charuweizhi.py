# -*- coding: utf-8 -*-
# 寻找插入位置 
import string 

line1 = raw_input()
line2 = raw_input()
line3 = raw_input()

line1 = string.atoi(line1)
data = line2.split()
line2 = []
for i in xrange(len(data)):
    line2.append(string.atoi(data[i]))
    

line3 = string.atoi(line3)
if line3 in line2:
    print line2.index(line3)
else:
    i = 0
    j = len(line2)
    if line3 > line2[-1]:
        print len(line2)
    if line3 < line2[0]:
        print 0
    else:
        
        while j-i > 1:
            mid = (i+j) /2
            if( line3 < line2[mid] ):
                j = mid
            if(line3 > line2[mid]):
                i = mid
        print j