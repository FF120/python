# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 10:41:24 2016

@author: FF120
"""
import numpy as np
import os
import nibabel as nib
from nilearn.input_data import NiftiMasker
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.feature_selection import SelectFromModel


from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold

import fmriUtils as fm  #自定义函数


#====================================================================
root = r"D:\data_processing\Python"
os.chdir(root)
label_path = root+'\design\label.npy'
empty_tr_path = root+'\design\empty_tr.npy'
mask_path = root + '\design\mask\WholeBrainMask.nii'
func_path = root + '\Sub001\wBoldImg4D_sub001.nii'

spmT_path = root + '\Sub001\spmT_0001.img'

label = np.load(label_path)
empty_tr = np.load(empty_tr_path)

nifti_masker = NiftiMasker(mask_img=mask_path, 
                           standardize=False,
                           memory="nilearn_cache", memory_level=1)               
X = nifti_masker.fit_transform(func_path)
X = np.delete(X,empty_tr-1,axis=0)
y = label

np.save('X.npy',X)
np.save('y.npy',y)

#==========================================
root = r"D:\data_processing\Python"
os.chdir(root)
X = np.load('X.npy')
y = np.load('y.npy')
#===============================================
from sklearn.linear_model import ElasticNet
y = fm.defineClass(y)
enet = ElasticNet(alpha=0.1, l1_ratio=0)
enet_fitted = enet.fit(X, y)
model = SelectFromModel(enet_fitted,prefit=True,threshold=0.00001)
XX = model.transform(X)














#====================================================================
cache_path=root + '\\rl_cache'
y = fm.defineClass(y)
clf_l1_LR = RandomizedLogisticRegression(C=0.1,tol=0.01,memory=cache_path)
clf_l1_LR.fit(X, y)
from sklearn.externals import joblib
joblib.dump(clf_l1_LR, 'RandomizedLogisticRegression.model')

#加载模型
# clf = joblib.load('filename.pkl') 
support_mask = clf_l1_LR.get_support
model = SelectFromModel(clf_l1_LR,prefit=True)
XX = model.transform(X)
print "=======LR model========="
print clf_l1_LR
img1 = X[0,:]
for i in xrange(len(img1)):
    if not support_mask[i]:
        img1[i] = 0
coef_img = nifti_masker.inverse_transform(img1)
coef_img.to_filename(r'D:\aaa.nii')
