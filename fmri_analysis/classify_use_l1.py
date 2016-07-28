# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 20:25:14 2016

@author: FF120
"""
import scipy.io as sio 
import numpy as np

from nilearn.input_data import NiftiMasker
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold

import fmriUtils as fm  #自定义函数

n_folds = 5

#f = fm.outTo() #输出重定向到文件

label_path = "D:/data_processing/jianlong/data_processing/mvpa/design/label.mat"
empty_tr_path = "D:/data_processing/jianlong/data_processing/mvpa/design/a.mat"
mask_path = "D:/data_processing/jianlong/data_processing/mvpa/design/allMask.nii"
func_filename = "D:/data_processing/Python/Sub002/wBoldImg4D_sub002.nii"

label_mat=sio.loadmat(label_path) 
empty_tr_mat = sio.loadmat(empty_tr_path)
label = label_mat['label']
label=label.reshape(-1,)
empty_tr=empty_tr_mat['a']
nifti_masker = NiftiMasker(mask_img=mask_path, 
                           standardize=True,
                           memory="nilearn_cache", memory_level=1)               
X = nifti_masker.fit_transform(func_filename)
X = np.delete(X,empty_tr-1,axis=0)
y = label

 
y = fm.defineClass(y)
#控制选择的特征的数量的参数
cc = [0.1]
score = []
print "======原始特征========="
print X.shape
print "*"*80
for c in cc:
    
    clf_l1_LR = LogisticRegression(C=c, penalty='l1', tol=0.01)
    clf_l1_LR.fit(X, y)
    coef = clf_l1_LR.coef_
    print "=======LR model========="
    print clf_l1_LR
    model = SelectFromModel(clf_l1_LR, prefit=True)
    feature_mask = model._get_support_mask() #获得特征选择的下标
    new_mask = feature_mask.astype('float64')
    coef_img = nifti_masker.inverse_transform(new_mask)
    coef_img.to_filename('D:\sub002_wholemask.nii')

    XX = model.transform(X)
    yy = y
    print "====新的特征======="
    print XX.shape
    cv = StratifiedKFold(yy,n_folds)
    cv_scores = []

    for train, test in cv:
        svc = SVC(kernel='linear')
        svc.fit(XX[train], yy[train])
        prediction = svc.predict(XX[test])
        cv_scores.append( np.sum(prediction == yy[test]) / float(np.size(yy[test])) )
    
    score.append(np.mean(cv_scores))
    print "=====分类准确率======="
    print cv_scores
    print np.mean(cv_scores)

print "=======分类准确率最高的参数取值======"
print cc[score.index(max(score))]


