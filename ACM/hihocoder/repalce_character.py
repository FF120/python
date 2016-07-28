# -*- coding: utf-8 -*-
'''
实现替换不等式中的符号，返回真假

例子：
A<B<=C
3<B<=10

输入A=1，B=4，C=10
输出：True
判断是否满足上面给出的两个式子，都满足返回真，否则返回假

思路：
python中的eval()可以执行字符串表达式，所以只需要将规则中的符号替换成
具体的数字，就可以使用eval()计算该表达式直接得到结果。
'''
import string
lineString = raw_input()
line = string.replace(lineString,'<=','=')
chk = {'A':42,'B':2,'C':3}
for i in range(len(chk)):
    keys = chk.keys()
    line = string.replace(line,str(keys[i]),str(chk[keys[i]]))

for i in range( len(line)-1 ):
    temp = []
    if line[i] == '<' or line[i] == '=':
        if line[i] == '<':
            if line[i-1] < line[i+1]:
                temp.append('T')
            else:
                temp.append('F')
        if line[i] == '=':
            if line[i-1] <= line[i+1]:
                temp.append('T')
            else:
                temp.append('F')
                
    if 'F' in line:
        print 'False'

print 'True'