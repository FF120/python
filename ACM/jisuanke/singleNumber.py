# -*- coding: utf-8 -*-

#����һ��list�����֮��

def xorList(s):
    if len(s) == 1:
        return s[0]
    sums = s[0] ^ s[1]

    for i in xrange(2,len(s)):
        sums = sums ^ s[i]
    
    return sums
"""
������ֻ��һ�����ֳ���1�Σ��������е����ֶ�����2�Σ��ҳ����ֻ����һ�ε�����

˼·��������ͬ���������֮��Ϊ0������������֮���ʣ��ֻ����һ�ε�����
"""
s = [1,1,1,1,33,2,2,3,3,4,4]
   
print xorList(s),s.index(xorList(s))
print "*" * 80

"""
������ֻ���������ֳ���1�Σ������������ֶ��������Σ��ҳ�������ֻ����
һ�ε�����

˼·�����������һ�Σ��õ�a^b��ֵ���ҳ�a��b�ĸ�������λ��һ����Ȼ��������������λ������
�е����ֳ�2�࣬�ö�����λΪ0��  �� �ö�����λΪ1�ġ�Ȼ��ֱ����
"""
s = [10,140,1,1,2,2,3,3,4,4,5,5]

sums = xorList(s)
import math
b = int( math.log(sums,2) ) + 1 #���λ��1
#���ݵ�bλ��0����1�ֳ�����
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
������ֻ��1�����ֳ���1�Σ��������е����ֶ�����3�Σ��ҳ����ֻ����һ�ε���
"""
s = [4,4,4,1,1,1,7,7,7,200,200,200,90]
import math
'''
����lists,���ֻ����1�ε����֣���������ֶ���������
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

������ֻ����������ֻ����һ�Σ���������ֶ��������Σ��ҳ�������ֻ����һ�ε�����
"""
