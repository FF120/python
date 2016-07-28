# -*- coding: utf-8 -*-
"""
使用递归特征消除选取特征
"""
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV

import numpy as np
import fmriUtils as fm

n_folds = 5

f = fm.outTo() #输出重定向到文件
X,y = fm.loadData2()   

y = fm.defineClass(y)

svc = SVC(kernel="linear")
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(y, n_folds),
              scoring='accuracy')
rfecv.fit(X, y)
print("Optimal number of features : %d" % rfecv.n_features_)