# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:48:48 2016

@author: FF120
"""

'''

用SVM获得每个特征对分类结果的贡献程度，按照贡献程度从大到小排名，选出贡献程度最大的
前K个特征作为特征选择的结果,使用SVM的时候，排名的依据是fit之后的coef_值。

这里的估计器可以替换成任何其他方法，如GLM
'''

from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import numpy as np
# Load the digits dataset
digits = load_digits()
X = digits.images.reshape((len(digits.images), -1))
y = digits.target
print "原来的特征："
print X.shape

# Create the RFE object and rank each pixel
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=10, step=1)
ref = rfe.fit(X, y)
print "选择的特征的个数"
print np.sum(ref._get_support_mask())
print ref._get_support_mask()
print rfe.ranking_



