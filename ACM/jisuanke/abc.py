
#A+B+C ����
import string
a = raw_input()  #��ȡ�û�����
ll = a.split()  #�ÿո�ָ��ַ���������list
s = 0
for num in ll:
    s += string.atoi(num)  #�ַ���ת����
print s
	