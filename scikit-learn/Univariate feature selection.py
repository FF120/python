# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 10:00:08 2016

@author: FF120
"""

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
#SelectKBest -- f_classif
from sklearn.feature_selection import f_classif
iris = load_iris()
X, y = iris.data, iris.target
X_fitted = SelectKBest(f_classif, k=3).fit(X,y)
print "SelectKBest -- f_classif"
print X_fitted.scores_
print X_fitted.pvalues_
print X_fitted.get_support()
X_transformed = X_fitted.transform(X)
print X_transformed.shape
#SelectKBest -- chi2
from sklearn.feature_selection import chi2
X_fitted_2 = SelectKBest(chi2, k=3).fit(X,y)
print "SelectKBest -- chi2"
print X_fitted_2.scores_
print X_fitted_2.pvalues_
print X_fitted_2.get_support()
X_transformed_2 = X_fitted_2.transform(X)
print X_transformed_2.shape

#SelectPercentile -- f_classif
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
X_fitted_3 = SelectPercentile(f_classif, percentile=50).fit(X,y)
print "SelectPercentile -- f_classif"
print X_fitted_3.scores_
print X_fitted_3.pvalues_
print X_fitted_3.get_support()
X_transformed_3 = X_fitted_3.transform(X)
print X_transformed_3.shape

#SelectPercentile -- chi2
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
X_fitted_4 = SelectPercentile(chi2, percentile=50).fit(X,y)
print "SelectPercentile -- chi2"
print X_fitted_4.scores_
print X_fitted_4.pvalues_
print X_fitted_4.get_support()
X_transformed_4 = X_fitted_4.transform(X)
print X_transformed_4.shape

#SelectFpr --- chi2
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import chi2
X_fitted_5 = SelectFpr(chi2, alpha=2.50017968e-15).fit(X,y)
print "SelectFpr --- chi2"
print X_fitted_5.scores_
print X_fitted_5.pvalues_
print X_fitted_5.get_support()
X_transformed_5 = X_fitted_5.transform(X)
print X_transformed_5.shape

#SelectFpr --- f_classif
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import f_classif
X_fitted_6 = SelectFpr(f_classif, alpha=1.66966919e-31 ).fit(X,y)
print "SelectFpr --- f_classif"
print X_fitted_6.scores_
print X_fitted_6.pvalues_
print X_fitted_6.get_support()
X_transformed_6 = X_fitted_6.transform(X)
print X_transformed_6.shape

# SelectFdr  和 SelectFwe 的用法和上面类似，只是选择特征时候的依据不同，真正决定得分不同的是
#统计检验方法，从上面可以看到，使用f_classif的得出的参数都相同。