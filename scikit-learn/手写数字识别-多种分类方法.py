# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 22:31:56 2016

@author: FF120
"""
import sys
import os
import numpy as np

from collections import Counter
from time import time
from data_utils import DataUtils

from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

reload(sys)
sys.setdefaultencoding('utf8')   #解决编码错误

program_start = time()
root_path = r"D:\FF120\workspace\Python\data\MINIST" #MINIST文件路径
save_to_disk = False #是否生成图像保存到硬盘
 
if os.path.exists(root_path + "\\train_X.npy"):
    train_X = np.load(root_path + '\\train_X.npy')
    train_y = np.load(root_path + '\\train_y.npy')
    test_X = np.load(root_path + '\\test_X.npy')
    test_y = np.load(root_path + '\\test_y.npy')
else:
    trainfile_X = root_path + '\\train-images.idx3-ubyte'
    trainfile_y = root_path + '\\train-labels.idx1-ubyte'
    testfile_X =  root_path + '\\t10k-images.idx3-ubyte'
    testfile_y = root_path + '\\t10k-labels.idx1-ubyte'
    
    train_X = DataUtils(filename=trainfile_X).getImage()
    train_y = DataUtils(filename=trainfile_y).getLabel()
    test_X = DataUtils(testfile_X).getImage()
    test_y = DataUtils(testfile_y).getLabel()
    
    np.save(root_path +"\\train_X.npy",train_X)
    np.save(root_path +"\\train_y.npy",train_y)
    np.save(root_path +"\\test_X.npy",test_X)
    np.save(root_path +"\\test_y.npy",test_y)

#以下内容是将图像保存到本地文件中
if save_to_disk:
    t0 = time.time()
    path_trainset = root_path + "\\imgs_train"
    path_testset = root_path + "\\imgs_test"
    if not os.path.exists(path_trainset):
        os.mkdir(path_trainset)
    if not os.path.exists(path_testset):
        os.mkdir(path_testset)
    DataUtils(outpath=path_trainset).outImg(train_X, train_y)
    DataUtils(outpath=path_testset).outImg(test_X, test_y)
    t1 = time.time() - t0
    print "生成图像耗时：%d s" % t1

#这里需要从文件中读取图像和标签，训练数据和测试数据，达到和上面train_X,train_Y, test_X,test_Y相同的效果。
#img=mpimg.imread('D:\\imgs_train\\0_5.png',as_grey=True)

## 数据分析，统计每个类有多少Sample，特征的取值都是什么样的，


## 输出重定向到文件
f=open(root_path + '\\result.txt','w')
import sys
old=sys.stdout #将当前系统输出储存到一个临时变量中
sys.stdout=f  #输出重定向到文件


train_X = train_X[0:600,:]
train_y = train_y[0:600]     ##截取前600分析
classes_num = Counter(train_y)
feature = np.unique(train_X)
label = np.unique(train_y)

test_X = test_X[0:100,:]
test_y = test_y[0:100] 
classes_num = Counter(test_y)
print('='*80)
print 'data analysis'
print('-'*80)
print "类别数量："
print classes_num
feature = np.unique(test_X)
print "特征的取值："
print feature
label = np.unique(test_y)
print "标签的取值："
print label
print('='*80)

# 通用训练和测设分类器
def benchmark(clf, parameters):
    print('-' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(train_X, train_y)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(test_X)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(test_y, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))
        
    clf_descr = str(clf).split('(')[0]
    print('-'*40)
    print('parameters searching')
    print('-'*40)
    if parameters:
        grids_search = GridSearchCV(clf, parameters)
        t0 = time()
        grids_search.fit(train_X, train_y)
        t1 = time() - t0
        print "optimal parameters:" 
        print grids_search.best_params_
        print "accuracy：" 
        print grids_search.best_score_
        print "waste time：%d s" % t1
        return clf_descr, grids_search.best_score_, train_time, test_time
    else:
        return clf_descr, score, train_time, test_time
        
results = []

#参数搜索条件
parameters_condition = {
    'RidgeClassifier':[{'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']}],
    'Perceptron':[{'penalty':['l1','l2','elasticnet'],'alpha':[0.00001,0.0001,0.001,0.01]}],
    'PassiveAggressiveClassifier':[{'C':[0.1,1.0,10],'n_iter':[2,5,8]}],
    'SGDClassifier':[{'penalty': ['l2'], 
                     'alpha': [0.00001,0.0001,0.001,0.01,0.1],
                     'loss':['hinge','squared_hinge','modified_huber','log']}],
    'KNeighborsClassifier':[{'n_neighbors': [1,2,3,4,5,6], 
                     'weights': ['uniform','distance'],
                     'algorithm':['ball_tree','kd_tree','brute','auto']}],
    'NearestCentroid':[],
    'SVC':[{'kernel': ['rbf'], 'gamma': [1e-2,1e-3],'C': [1,2,3,4,5,10]},
            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}],
    'LinearSVC':[],
    'NuSVC':[{'gamma':[1e-2,1e-3,'auto']}],
    'MultinomialNB':[{'alpha':[0.1,1.0,10]}],
    'BernoulliNB':[{'alpha':[0.1,1.0,10],'fit_prior':[False,True]}],
    'RandomForestClassifier':[{'n_estimators':[5,10,15,20],'min_samples_split':[2,3,4]}]
}

#训练和测试
for clf, name, parameters in (
        (RidgeClassifier(), "Ridge Classifier of linear_model",parameters_condition['RidgeClassifier']),
        (Perceptron(), "Perceptron of linear_model",parameters_condition['Perceptron']),
        (PassiveAggressiveClassifier(), "Passive-Aggressive of linear_model",parameters_condition['PassiveAggressiveClassifier']),
        (SGDClassifier(), "SGD model of linear_model",parameters_condition['SGDClassifier']),   
        (neighbors.KNeighborsClassifier(),"kNN",parameters_condition['KNeighborsClassifier']),
        (neighbors.NearestCentroid(),"NearestCentroid",parameters_condition['NearestCentroid']),
        (svm.SVC(), "SVC",parameters_condition['SVC']),
        (svm.LinearSVC(), "LinearSVC",parameters_condition['LinearSVC']),
        (svm.NuSVC(), "NuSVC",parameters_condition['NuSVC']),
        (MultinomialNB(),"MultinomialNB",parameters_condition['MultinomialNB']),
        (BernoulliNB(),"BernoulliNB",parameters_condition['BernoulliNB']),
        (RandomForestClassifier(), "Random forest",parameters_condition['RandomForestClassifier'])):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf=clf, parameters=parameters))



# make some plots
import matplotlib.pyplot as plt

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

spend_time = time() - program_start
print('*'*80)
print "total runtime: %d s" % spend_time
print('*'*80)

## 输出重定向回来
sys.stdout=old #还原原系统输出
f.close() 
###================================SVM==========================================
###
##'''
##RandomizedSearchCV 随机搜索参数
#import scipy
#n_iter_search = 20
#param_grid = {"kernel": ['rbf'],
#              "gamma": scipy.stats.expon(scale=0.1),
#              "C": scipy.stats.expon(scale=100)}
#              
#random_search = RandomizedSearchCV(svm.SVC(C=1), param_distributions=param_grid,
#                                   n_iter=n_iter_search)
#start = time.time()
#random_search.fit(train_X, train_y)
#print("RandomizedSearchCV took %.2f seconds for %d candidates"
#      " parameter settings." % ((time.time() - start), n_iter_search))
#
#random_search.grid_scores_
#random_search.best_estimator_
#random_search.best_score_
#random_search.best_params_
#'''
#random_search.best_estimator_
#Out[32]: 
#SVC(C=7.81231733242, cache_size=200, class_weight=None, coef0=0.0,
#  decision_function_shape=None, degree=3, gamma=0.0166642325028,
#  kernel='rbf', max_iter=-1, probability=False, random_state=None,
#  shrinking=True, tol=0.001, verbose=False)
#
#random_search.best_score_
#Out[33]: 0.88500000000000001
#
#random_search.best_params_
#Out[34]: {'C': 7.812317332417196, 'gamma': 0.016664232502838144, 'kernel': 'rbf'}
#
#'''






