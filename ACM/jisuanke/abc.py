
#A+B+C 问题
import string
a = raw_input()  #获取用户输入
ll = a.split()  #用空格分割字符串，返回list
s = 0
for num in ll:
    s += string.atoi(num)  #字符串转数字
print s
	