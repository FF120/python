# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 10:24:31 2016

@author: FF120
"""
import sys

while True:
    num = sys.stdin.readline()[:-1] # -1 to discard the '\n' in input stream
    add = num.split(' ')
    s = 0
    for a in add:
        s+=int(a)
        
    print s