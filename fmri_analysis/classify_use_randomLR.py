# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 20:25:14 2016

@author: FF120
"""
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import RandomizedLogisticRegression
import fmriUtils as fm  #自定义函数

n_folds = 10

f = fm.outTo() #输出重定向到文件
X,y = fm.loadData2()   
X2,y2 = fm.loadData2()   

y = fm.defineClass(y)

randomized_logistic = RandomizedLogisticRegression(C=0.1,n_jobs=2)
randomized_logistic.fit(X,y)
XX = randomized_logistic.transform(X)
print "============选择后剩余的特征================"
print XX.shape

yy = y
cv = StratifiedKFold(yy,n_folds)
cv_scores = []
for train, test in cv:
    svc = SVC(kernel='linear')
    svc.fit(XX[train], yy[train])
    prediction = svc.predict(XX[test])
    cv_scores.append( np.sum(prediction == yy[test]) / float(np.size(yy[test])) )
    
print "========分类准确率======="
print cv_scores,np.mean(cv_scores)
