# -*- coding: utf-8 -*-
'''
ʵ���滻����ʽ�еķ��ţ��������

���ӣ�
A<B<=C
3<B<=10

����A=1��B=4��C=10
�����True
�ж��Ƿ������������������ʽ�ӣ������㷵���棬���򷵻ؼ�

˼·��
python�е�eval()����ִ���ַ������ʽ������ֻ��Ҫ�������еķ����滻��
��������֣��Ϳ���ʹ��eval()����ñ��ʽֱ�ӵõ������
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