'''

批量计算某个目录下的所有文件的MD5，用于比较文件是否一致
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
     """ 计算文件的MD5值
     """
     def read_chunks(fh):
         fh.seek(0)
         chunk = fh.read(8096)
         while chunk:
             yield chunk
             chunk = fh.read(8096)
         else: #最后要将游标放回文件开头
             fh.seek(0)
     m = hashlib.md5()
     if isinstance(fname, basestring) \
             and os.path.exists(fname):
         with open(fname, "rb") as fh:
             for chunk in read_chunks(fh):
                 m.update(chunk)
     #上传的文件缓存 或 已打开的文件流
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


##比较两个文件的不同
import difflib
import sys
 
a = open('a.txt', 'U').readlines()
b = open('b.txt', 'U').readlines()
diff = difflib.ndiff(a, b)
 
sys.stdout.writelines(diff)  