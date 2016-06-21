# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 20:13:51 2016

@author: FF120
"""
from sklearn import svm
X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]
# "one-against-one"
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X, Y) 
'''
one-against-one 就是一对一，假设这四类的名称为a,b,c,d.
则需要训练区分(a,b)(a,c)(a,d)(b,c)(b,d)(c,d)的6种模型，所以
one-against-one这种策略在做多分类问题的时候会生成n*(n-1)/2个模型，每个模型区分其中的两个类。
'''
dec = clf.decision_function([[1]])
dec.shape[1] # 4 classes: 4*3/2 = 6
'''
 "one-vs-the-rest" 就是一对余下所有的，假设四类的名称为a,b,c,d;
 则需要训练区分(a,bcd),(b,acd)(c,abd)(d,abc)的4种模型，每个模型区分其中一个类，被除此类之外的所有类当作另外一个类处理。
 这种策略在做多分类问题的时候会生成n个模型。
'''

clf.decision_function_shape = "ovr"
dec = clf.decision_function([[1]])
dec.shape[1] # 4 classes

