'''

��������ĳ��Ŀ¼�µ������ļ���MD5�����ڱȽ��ļ��Ƿ�һ��
'''

#!/usr/bin/env python
 #coding : utf-8
import hashlib
import os
def md5hex(word):
    if isinstance(word, unicode):
         word = word.encode("utf-8")
    elif not isinstance(word, str):
         word = str(word)
    m = hashlib.md5()
    m.update(word)
    return m.hexdigest()
 
def md5sum(fname):
     """ �����ļ���MD5ֵ
     """
     def read_chunks(fh):
         fh.seek(0)
         chunk = fh.read(8096)
         while chunk:
             yield chunk
             chunk = fh.read(8096)
         else: #���Ҫ���α�Ż��ļ���ͷ
             fh.seek(0)
     m = hashlib.md5()
     if isinstance(fname, basestring) \
             and os.path.exists(fname):
         with open(fname, "rb") as fh:
             for chunk in read_chunks(fh):
                 m.update(chunk)
     #�ϴ����ļ����� �� �Ѵ򿪵��ļ���
     elif fname.__class__.__name__ in ["StringIO", "StringO"] \
             or isinstance(fname, file):
         for chunk in read_chunks(fname):
             m.update(chunk)
     else:
         return ""
     return m.hexdigest()
     
print md5sum('D:/Share.txt')
root = "D:/data_processing/jianlong/data_processing/img_hdr/20160713001/ep2d_bold_moco_p2_rest_0006/"
ll = os.listdir(root)
fp = open("md5sum.txt",'w')   
for l in ll:
    str = md5sum(root+l) + "    " + l + '\n'
    fp.write(str)   

fp.close()      


##�Ƚ������ļ��Ĳ�ͬ
import difflib
import sys
 
a = open('a.txt', 'U').readlines()
b = open('b.txt', 'U').readlines()
diff = difflib.ndiff(a, b)
 
sys.stdout.writelines(diff)  