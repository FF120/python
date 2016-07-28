#简单斐波那契 

import string

a = raw_input()
line = a.split()
num = string.atoi(line[0])

lists = [0,1]
for i in xrange(num):
    lists.append(lists[-1] + lists[-2])
    
print lists[num]