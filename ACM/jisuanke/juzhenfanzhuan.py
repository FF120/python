# 矩阵翻转 
# -*- coding: utf-8 -*-
import sys
import string

a = raw_input()
line = a.split()
M = string.atoi(line[0])
N = string.atoi(line[1])
T = string.atoi(line[2])

lists = []

for i in xrange(M):
   a = raw_input()
   a = a.split()
   temp = []
   for j in xrange(N):
       temp.append( string.atoi( a[j] ) )
   lists.append(temp)

    
if T == 0: #左右翻转
    for i in range(M):
        for j in range(N)[::-1]: # 倒序输出
            sys.stdout.write(str(lists[i][j]))
            sys.stdout.write(" ")
        print ""
if T == 1: #上下翻转
    for i in range(M)[::-1]:
        for j in range(N): # 倒序输出
            sys.stdout.write(str(lists[i][j]))
            sys.stdout.write(" ")
        print ""