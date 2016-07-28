# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 18:24:10 2016

@author: FF120
"""
import numpy as np
from time import time
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.cross_validation import StratifiedKFold


import fmriUtils as fm
n_folds = 10

f = fm.outTo() #输出重定向到文件
X,y = fm.loadData2()   
y = fm.defineClass(y)
t0 = time()
forest = ExtraTreesClassifier(n_estimators=1000,
                              max_features=1000,
                              n_jobs=2,
                              random_state=0)
                              
forest.fit(X, y)
model = SelectFromModel(forest, threshold='2*mean',prefit=True)
XX = model.transform(X)

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
