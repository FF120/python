# -*- coding: utf-8 -*-
"""
记录一些常用的语句
"""


#保存numpy
import numpy as np
label = np.array([1,2,3,4,5,6,7,8,9,0])
np.save('label.npy',label) #保存numpy变量


#改变当前目录
import os
path = r"D:\data_processing\Python"
os.chdir(path)

# 加载fmri图像数据
import nibabel as nib
path = r"d:\c2s20160713001-193508-00001-00176-1.img"
img = nib.load(path)
img_data = img.get_data()

#保存处理之后的fmri图像
from nilearn.input_data import NiftiMasker
nifti_masker = NiftiMasker(mask_img='D:\mask.img', 
                           standardize=False,
                           memory="nilearn_cache", memory_level=1)               
X = nifti_masker.fit_transform('D:\funcImg.nii')
img1 = np.array([1,1,1,0,0,0,0,0])  #img 是和mask.img一样大小的一个矩阵
coef_img = nifti_masker.inverse_transform(img1)
coef_img.to_filename(r'D:\aaa.nii')


# scikit-learn 模型持久化
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
clf_l1_LR = LogisticRegression(C=0.1, penalty='l1', tol=0.01)
joblib.dump(clf_l1_LR, 'LogisticRegression.model')


#加载scikit-learn模型
from sklearn.externals import joblib
clf = joblib.load('filename.pkl') 

