# 判断质数
import string
import math

a = raw_input()
ll = a.split()
num = string.atoi(ll[0])
max_index = int( math.sqrt(num) )
flag = True
for i in xrange(2,max_index):
    if num % i == 0:
        flag = False
        print "NO"
        break
if flag:
    print "YES"