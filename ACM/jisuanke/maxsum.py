# -*- coding: utf-8 -*-
# 最大子序列和
import string
import sys
line = raw_input()
line2 = raw_input().split()
nums = []
for num in line2:
    nums.append(string.atoi(num))

thisSum = 0
MaxSum = 0
for i in xrange(0,len(nums)):
    thisSum += nums[i]
    if thisSum > MaxSum:
        MaxSum = thisSum
    elif thisSum < 0 :
        thisSum = 0
        
print MaxSum