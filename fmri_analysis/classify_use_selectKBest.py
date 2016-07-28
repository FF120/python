# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 09:42:44 2016

@author: FF120

使用scikit-learn的selectKBest方法选择特征
"""

import numpy as np

from nilearn import image
from nilearn.plotting import plot_stat_map, show


from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import LeaveOneLabelOut
from sklearn.cross_validation import permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from nilearn.input_data import NiftiMasker
import numpy as np


import fmriUtils as fm

n_folds = 10
f=fm.outTo()
X,y = fm.loadData()
y = fm.defineClass(y)

XX = X
yy = y
#使用SVM分类和预测
svc = SVC(kernel='linear')
#
feature_selection = SelectKBest(f_classif, k=1000)
anova_svc = Pipeline([('anova', feature_selection), ('svc', svc)])

cv = StratifiedKFold(yy, n_folds=n_folds)

cv_scores = []

for train, test in cv:
    anova_svc.fit(XX[train], yy[train])
    prediction = anova_svc.predict(XX[test])
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

