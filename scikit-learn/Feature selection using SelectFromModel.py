# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 16:53:23 2016

@author: FF120
"""

print(__doc__)

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

# Load the boston dataset.
boston = load_boston()
X, y = boston['data'], boston['target']

# We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
clf = LassoCV()
clf.fit(X,y)
# Set a minimum threshold of 0.25
sfm = SelectFromModel(clf, threshold='mean',prefit=True)
print X.shape
#sfm = sfm.fit(X, y)
print "============LassoCV================"
print "选择的特征"
print sfm._get_support_mask();
n_features = sfm.transform(X).shape[1]
print n_features

# We use LinearSVC
from sklearn.svm import LinearSVC
#C 越小，选择的特征越少
lsvc = LinearSVC(C=0.001, penalty="l1", dual=False)
y = y.astype(np.int64) #转换成整数，因为是分类器，不是回归
lsvc.fit(X,y)
model = SelectFromModel(lsvc, prefit=True)
print "============线性SVM==============================="
print "选择的特征"
print model._get_support_mask();
n_features = model.transform(X).shape[1]
print n_features


from sklearn import linear_model
clf = linear_model.LogisticRegression(C=0.001, penalty='l1', tol=1e-6)
y = y.astype(np.int64) #转换成整数，因为是分类器，不是回归
clf.fit(X,y)
model = SelectFromModel(clf, prefit=True)
print "============逻辑回归==============================="
print "选择的特征"
print model._get_support_mask();
n_features = model.transform(X).shape[1]
print n_features

from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier()
y = y.astype(np.int64) #转换成整数，因为是分类器，不是回归
clf = clf.fit(X, y)
model = SelectFromModel(clf, prefit=True)
print "============基于树的特征选择==============================="
print clf.feature_importances_ 
print "选择的特征："
print model._get_support_mask();
n_features = model.transform(X).shape[1]
print n_features