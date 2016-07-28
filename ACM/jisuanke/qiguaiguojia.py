# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:20:05 2016
 奇怪的国家 
@author: FF120
"""
# -*- coding: utf-8 -*-


import sys

line1 = raw_input()
line2 = raw_input()
def myAnd(x,y):
    if x == '1' and y == '1':
        return '1'
    if x !=y :
        return '0'
    if x == '0' and  y == '0':
        return '1'
   
for i in xrange(len(line1)):
    sys.stdout.write(myAnd(line1[i],line2[i]))
    

    




