# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 15:30:51 2016

@author: FF120
"""
#将【1，2，3，4，5】这样的list输出成 1，2，3，4，5这样的string
def list_to_str(list):
    str1 = str(list)
    str1 = str1.replace(']','').replace('[','').replace(' ','')
    return str1

for line in open('e:/test_sigmoid222.txt','r'):
    aa =  line.strip('\n')  .split('\t');
    bb = map(int,aa[1].split(','));
    cc = []
    maxValues = max(bb)
    minValues = min(bb)
    for x in bb:
        y = (float)(x-minValues)/(maxValues-minValues)
        y = (int)(y*1000)
        cc.append(y)
    with open('e:/test_sigmoid.txt','a') as of:
        outstr = aa[0]
        outstr = outstr + "\t"
        outstr = outstr + list_to_str(cc)
        outstr = outstr + "\n"
        of.write(outstr)
