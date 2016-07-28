# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:05:41 2016

@author: FF120
"""
import os
import numpy as np
import fmriUtils as fm
from matplotlib import pyplot as plt
from matplotlib import font_manager

root = r"D:\data_processing\Python"
os.chdir(root)
X = np.load('X.npy')
y = np.load('y.npy')
y = fm.defineClass(y)

zhfont = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/truetype/arphic/ukai.ttc')

""" 比较好看的绘制方法 """ 
plt.figure(figsize=(8, 5), dpi=80)
axes = plt.subplot(111)

type1_x = []
type1_y = []
type2_x = []
type2_y = []
type3_x = []
type3_y = []
type4_x = []
type4_y = []

print 'range(len(labels)):'
print range(len(y))
for i in range(len(y)):
    if y[i] == 1:  # 不喜欢
        type1_x.append(X[i,:])
        type1_y.append(X[i][1])
    if y[i] == 2:  # 魅力一般
        type2_x.append(X[i][0])
        type2_y.append(X[i][1])
     if y[i] == 3: # 极具魅力 
         print i, '：', labels[i], ':', type(labels[i]) 
         type3_x.append(X[i][0]) 
         type3_y.append(X[i][1])
         
type1 = axes.scatter(type1_x, type1_y, s=20, c='red')
type2 = axes.scatter(type2_x, type2_y, s=40, c='green')
type3 = axes.scatter(type3_x, type3_y, s=50, c='blue')

plt.xlabel(u'每年获取的飞行里程数', fontproperties=zhfont)
plt.ylabel(u'玩视频游戏所消耗的事件百分比', fontproperties=zhfont)
axes.legend((type1, type2, type3), (u'不喜欢', u'魅力一般', u'极具魅力'), loc=2, prop=zhfont)
plt.show()



