# -*- coding: utf-8 -*-
"""
Created on Wed Jul 06 10:44:35 2016

@author: FF120
"""

'''
画线
'''
import matplotlib.pyplot as plt
x=[2,3,11]
y=[12,20,3]
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes # left, bottom, width, height (range 0 to 1)
axes2 = fig.add_axes([0.9, 0.1, 0.4, 0.3]) # inset axes
axes.plot(x,y,'r')
axes.set_xlabel('xx')
axes.set_ylabel('yy')
axes.set_title('title')

plt.show()

'''

离散点分布
'''
import matplotlib.pyplot as plt


N = 3
x=[2,3,11]
y=[12,20,3]
colors =  ('red', 'blue', 'lightgreen')
markers = ('s', 'o', 'x', '^', 'v')

fig = plt.figure()
#plt.xlim(0, 1)  # 设置坐标轴的范围
plt.scatter(x, y, c=colors, marker=markers[1],alpha=0.5)
plt.show()