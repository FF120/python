# -*- coding: utf-8 -*-
#泥塑课 
import string


while True:
    a = raw_input()
    line = a.split()
    num = string.atoi(line[0])
    if num == -1:
        break
    
    temp = {}
    for i in xrange(num):
        line = raw_input()
        line = line.split()
        a = string.atoi(line[0])
        b = string.atoi(line[1])
        c = string.atoi(line[2])
        name = line[3]
        temp[a*b*c] = name
    lists = temp.keys()
    name1 = temp.get(max(lists))
    name2 = temp.get(min(lists))
    print "%s took clay from %s" % (name1,name2)