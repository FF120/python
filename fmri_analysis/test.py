# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 10:41:24 2016

@author: FF120
"""

#=====================引入需要的库================================================
import numpy as np
import os
import nibabel as nib
from nilearn.input_data import NiftiMasker
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.feature_selection import SelectFromModel


from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold

import fmriUtils as fm  #自定义函数

#======================定义一些需要使用的常量=================================

root = r"D:\data_processing\Python"

#===================准备标签和特征============================================

os.chdir(root)
label_path = root+'\design\label.npy'
empty_tr_path = root+'\design\empty_tr.npy'
mask_path = root + '\design\mask\mask.img'
func_path = root + '\Sub001\wBoldImg4D_sub001.nii'
label = np.load(label_path)
empty_tr = np.load(empty_tr_path)

nifti_masker = NiftiMasker(mask_img=mask_path, 
                           standardize=True,
                           memory="nilearn_cache", memory_level=1)               
X = nifti_masker.fit_transform(func_path)
X = np.delete(X,empty_tr-1,axis=0)
y = label

np.save('X.npy',X)
np.save('y.npy',y)

#================加载标签和特征==========================

os.chdir(root)
X = np.load('X.npy')
y = np.load('y.npy')

#================特征可视化显示==========================
XX_show = XX1
mean=[]
var=[]
for i in xrange(XX_show.shape[1]):
    mean.append( np.mean(XX_show[:,i]) )
    var.append( np.var(XX_show[:,i]) )
    
varmax = max(var)
varmin = max(var)
import matplotlib.pyplot as plt
fig = plt.figure(2)
plt.subplot(211)
#plt.xlabel('Voxel')
#plt.ylabel('Variance')
plt.title('Variance of Voxel')
#plt.grid(True)
#plt.xlim()
#plt.ylim(varmin,varmax)
xx = np.linspace(1,XX_show.shape[1],XX_show.shape[1])
plt.plot(xx,var,'r') 

plt.subplot(212)
plt.plot(xx,mean,'b')
plt.show()

#===================移除所有方差很低的特征=================================
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=2000)
X2 = sel.fit_transform(X)
support_mask = sel._get_support_mask()
#mask_img = X[1,:]
mask_img = support_mask.astype('float64')
#for i in xrange(support_mask.shape[0]):
#    if not support_mask[i]:
#        mask_img[i] = 0
        
coef_img = nifti_masker.inverse_transform(mask_img)
coef_img.to_filename(r'D:\aaa.img')

#==================单变量特征选择方法===========================
from sklearn.feature_selection import SelectPercentile, f_classif
XX1 = X
yy1 = fm.defineClass(y)
'''数据标准化'''
#from sklearn import preprocessing
#scaler = preprocessing.StandardScaler()
#scaler = scaler.fit(XX1)
#XX1 = scaler.transform(XX1)

selector = SelectPercentile(f_classif, percentile=10)
selector.fit(XX1, yy1)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
import matplotlib.pyplot as plt
fig = plt.figure(2)
plt.subplot(211)
plt.title('p-value of Voxel')
xx = np.linspace(1,XX1.shape[1],XX1.shape[1])
plt.plot(xx,scores,'r') 
plt.show()

support_mask = selector._get_support_mask()
mask_img = support_mask.astype('float64')
coef_img = nifti_masker.inverse_transform(mask_img)
coef_img.to_filename(r'D:\aaa.img')
XX1_new = selector.transform(XX1)

#==============特征之间的相关性====================================
import numpy as np
from scipy.stats import pearsonr
XX2 = X
aa = pearsonr(XX2[0,:],XX2[1,:])

#============L1正则=====================
from sklearn.linear_model import LogisticRegression
XX3 = X
clf_l1_LR = LogisticRegression(C=1, penalty='l1', tol=0.001)
clf_l1_LR.fit(X, y)
coef = clf_l1_LR.coef_
print "=======LR model========="
print clf_l1_LR
model = SelectFromModel(clf_l1_LR, prefit=True)
feature_mask = model._get_support_mask() #获得特征选择的下标
new_mask = feature_mask.astype('float64')
coef_img = nifti_masker.inverse_transform(new_mask)
coef_img.to_filename('D:\sub002_wholemask.img')
XX3_new = model.transform(XX3)

#=================================================