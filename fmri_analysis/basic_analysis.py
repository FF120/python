# -*- coding: utf-8 -*-
"""
基本分类

使用未经过特征选择的原始数据，跑几个分类算法
"""
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import fmriUtils as fm
from time import time

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_iris
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.utils.extmath import density
from sklearn import metrics

root = r"D:\data_processing\Python"
n_folds = 5

os.chdir(root)
X = np.load('X.npy')
y = np.load('y.npy')
y = fm.defineClass(y,according='noise')
f = fm.outTo()

#===========特征选择========================
clf_l1_LR = LogisticRegression(C=0.1, penalty='l1', tol=0.001)
clf_l1_LR.fit(X, y)
print clf_l1_LR
model = SelectFromModel(clf_l1_LR, prefit=True)
X = model.transform(X)
     

results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (LinearSVC(loss='l2', penalty='l2',dual=False, tol=1e-3), "Liblinear model with l2"),
        (LinearSVC(loss='l2', penalty='l1',dual=False, tol=1e-3), "Liblinear model with l1"),
        (SGDClassifier(alpha=.0001, n_iter=50,penalty='l2'), "SGD model with l2"),
        (SGDClassifier(alpha=.0001, n_iter=50,penalty='l1'), "SGD model with l1"),        
        (SGDClassifier(alpha=.0001, n_iter=50,penalty="elasticnet"),"SGD model with Elastic-Net penalty" ),       
        (NearestCentroid(), "NearestCentroid without threshold"),
        #(MultinomialNB(alpha=.01),"MultinomialNB"),
        #(BernoulliNB(alpha=.01),"BernoulliNB"),
        (RandomForestClassifier(n_estimators=100), "Random forest"),
        (Pipeline([('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False, tol=1e-3))),('classification', LinearSVC())]),"LinearSVC with L1-based feature selection"),
        ):
    print('=' * 80)
    print(name)
    results.append(fm.benchmarkWithCV(clf,X,y,n_folds))

# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='r')
plt.barh(indices + .3, training_time, .2, label="training time", color='g')
plt.barh(indices + .6, test_time, .2, label="test time", color='b')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()
