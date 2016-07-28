# -*- coding: utf-8 -*-

#计算一个list的异或之和

def xorList(s):
    if len(s) == 1:
        return s[0]
    sums = s[0] ^ s[1]

    for i in xrange(2,len(s)):
        sums = sums ^ s[i]
    
    return sums
"""
数组中只有一个数字出现1次，其余所有的数字都出现2次，找出这个只出现一次的数字

思路：两个相同的数字异或之后为0，这个数组异或之后就剩下只出现一次的数了
"""
s = [1,1,1,1,33,2,2,3,3,4,4]
   
print xorList(s),s.index(xorList(s))
print "*" * 80

"""
数组中只有两个数字出现1次，其余所有数字都出现两次，找出这两个只出现
一次的数字

思路，先整体异或一次，得到a^b的值，找出a和b哪个二进制位不一样，然后根据这个二进制位把数组
中的数分成2类，该二进制位为0的  和 该二进制位为1的。然后分别异或
"""
s = [10,140,1,1,2,2,3,3,4,4,5,5]

sums = xorList(s)
import math
b = int( math.log(sums,2) ) + 1 #最高位是1
#根据第b位是0还是1分成两组
s1 = []
s2 = []
for a in s:
    if a>>(b-1)&1 == 1:
        s1.append(a)
    else:
        s2.append(a)

print xorList(s1),s.index(xorList(s1))
print xorList(s2),s.index(xorList(s2))
print "*" * 80
"""
数组中只有1个数字出现1次，其余所有的数字都出现3次，找出这个只出现一次的数
"""
s = [4,4,4,1,1,1,7,7,7,200,200,200,90]
import math
'''
输入lists,输出只出现1次的数字，其余的数字都出现三次
'''
def threeLists(s):
    
    max_num = max(s)
    length = int( math.log(max_num,2) ) + 1

    ones = [0] * length
    twos = [0] * length
    for a in s:
        a_len = int(math.log(a,2)) + 1
    
        for i in xrange(a_len):
            if (a >> i) & 1 == 1:
                if ones[i] == 0:
                    ones[i] = 1
                elif ones[i] == 1:
                    if twos[i] == 1:
                        ones[i] = 1
                        twos[i] = 0
                    elif twos[i] == 0:
                        twos[i] = 1
                        ones[i] = 0

    for i in xrange(length):
        if ones[i] == 1 and twos[i] == 1:
            ones[i] = 0
            twos[i] = 0  
    
    num = 0
    for i in xrange(length):
        num = num + (2**i) * ones[i]      
    return num

print threeLists(s)   
print "*" * 80
"""

数组中只有两个数字只出现一次，其余的数字都出现三次，找出这两个只出现一次的数字
"""
