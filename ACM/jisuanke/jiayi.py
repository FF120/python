# -*- coding: utf-8 -*-
# 加一 
import string
import sys
line = raw_input()
line2 = raw_input().split()
nums = []
for num in line2:
    nums.append(string.atoi(num))

i = 1
flag = True
while i <= len(nums):
    num = nums[-i] + 1
    if(num > 9): 
        nums[-i] = num % 10
        if i == len(nums):
            flag = False
            sys.stdout.write('1')
            sys.stdout.write(' ')
            for num in nums:
               sys.stdout.write(str(num))
               sys.stdout.write(' ') 
            
        i = i + 1
    else:
        nums[-i] = num
        break
if flag:
    for num in nums:
        sys.stdout.write(str(num))
        sys.stdout.write(' ') 
 




