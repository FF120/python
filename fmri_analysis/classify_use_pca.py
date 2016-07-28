# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 15:11:07 2016

@author: FF120
"""
import numpy as np

from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.dummy import DummyClassifier

from sklearn.decomposition import PCA

import fmriUtils as fm

n_folds = 10
f = fm.outTo()
X,y = fm.loadData2()
y = fm.defineClass(y)

#使用SVM分类和预测
svc = SVC(kernel='linear')
#选择99%信息量的特征
pca = PCA(n_components=0.99)
XX = pca.fit_transform(X)
yy = y
print "========新的特征========"
print XX.shape

cv = StratifiedKFold(yy,n_folds)

cv_scores = []

for train, test in cv:
    svc.fit(XX[train], yy[train])
    prediction = svc.predict(XX[test])
    cv_scores.append( np.sum(prediction == yy[test]) / float(np.size(yy[test])) )
    
    
classification_accuracy = np.mean(cv_scores)
print "=============SVM==================="
print svc
print cv_scores
print classification_accuracy

null_cv_scores = cross_val_score(DummyClassifier(), XX, yy, cv=cv)  
print "==========随机分类准确率============"
print null_cv_scores
print np.mean(null_cv_scores)

