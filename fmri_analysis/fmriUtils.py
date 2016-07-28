# -*- coding: utf-8 -*-
import scipy.io as sio 
import nibabel as nib
import numpy as np
import sys,os

import nibabel as nib 

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.dummy import DummyClassifier
from nilearn.input_data import NiftiMasker
from sklearn.decomposition import PCA


'''
加载被试功能像，使用nifti_masker和全脑mask将数据格式转换成[n_sample,n_features]
返回特征和标签
'''
def loadData():
    label_path = "D:/data_processing/jianlong/data_processing/mvpa/design/label.mat"
    empty_tr_path = "D:/data_processing/jianlong/data_processing/mvpa/design/a.mat"
    mask_path = "D:/data_processing/jianlong/data_processing/mvpa/design/allMask.nii"
    func_filename = "D:/data_processing/Python/Sub001/wBoldImg4D_sub001.nii"
    
    label_mat=sio.loadmat(label_path) 
    empty_tr_mat = sio.loadmat(empty_tr_path)
    label = label_mat['label']
    label=label.reshape(-1,)
    empty_tr=empty_tr_mat['a']
    nifti_masker = NiftiMasker(mask_img=mask_path, 
                               standardize=True,
                               memory="nilearn_cache", memory_level=1)               
    X = nifti_masker.fit_transform(func_filename)
    XX = np.delete(X,empty_tr-1,axis=0)
    return XX,label,nifti_masker
  

'''
不使用外部模版，使用计算出来的模版
'''
def loadData2():
    label_path = "D:/data_processing/jianlong/data_processing/mvpa/design/label.mat"
    empty_tr_path = "D:/data_processing/jianlong/data_processing/mvpa/design/a.mat"
    func_filename = "D:/data_processing/Python/Sub001/wBoldImg4D_sub001.nii"
    
    label_mat=sio.loadmat(label_path) 
    empty_tr_mat = sio.loadmat(empty_tr_path)
    label = label_mat['label']
    label=label.reshape(-1,)
    empty_tr=empty_tr_mat['a']
    nifti_masker = NiftiMasker(standardize=True,mask_strategy='epi',
                               memory="nilearn_cache", memory_level=1)               
    X = nifti_masker.fit_transform(func_filename)
    XX = np.delete(X,empty_tr-1,axis=0)
    return XX,label  
    
def loadData_sub002():
    label_path = "D:/data_processing/jianlong/data_processing/mvpa/design/label.mat"
    empty_tr_path = "D:/data_processing/jianlong/data_processing/mvpa/design/a.mat"
    func_filename = "D:/data_processing/Python/Sub002/wBoldImg4D_sub002.nii"
    
    label_mat=sio.loadmat(label_path) 
    empty_tr_mat = sio.loadmat(empty_tr_path)
    label = label_mat['label']
    label=label.reshape(-1,)
    empty_tr=empty_tr_mat['a']
    nifti_masker = NiftiMasker(standardize=True,mask_strategy='epi',
                               memory="nilearn_cache", memory_level=1)               
    X = nifti_masker.fit_transform(func_filename)
    XX = np.delete(X,empty_tr-1,axis=0)
    return XX,label 
    
'''
初始加载的标签是12种类别，这里返回需要的类别
'''
def defineClass(label,according="class"):
    if according == "class":
        for i in xrange(label.shape[0]):
            if label[i] in [1,2,3,4]:
                label[i] = 1
            elif label[i] in [5,6,7,8]:
                label[i] = 2
            elif label[i] in [9,10,11,12]:
                label[i] = 3
            elif label[i] in [13,14,15,16]:
                label[i] = 4
                
        return label
    if according == "noise":
        for i in xrange(label.shape[0]):
            if label[i] in [1,3,5,7,9,11,13,15]:
                label[i] = 1
            elif label[i] in [2,4,6,8,10,12,14,16]:
                label[i] = 2
                
        return label
        
'''
输出重定向到文件
'''
def outTo(filename="D:\out.txt"):
    current_path = os.path.abspath('.')
    os.chdir(current_path) 
    filename = sys.argv[0].split("\\")[0]+str('.txt');
    f=open(filename,'w')
    sys.stdout=f  #输出重定向到文件
    return f
    
'''
显示图像
'''
def showImg(filename):
    img = nib.load(filename)
    img_data = img.get_data()
    img_head = img.header
    affine_array = img.affine
    print "=======图像信息================="
    print img.shape
    print img_data.shape
    print img_head.get_data_shape()
    print img_head.get_zooms()
    print affine_array
    
#    from nilearn import image
#    from nilearn.plotting import plot_stat_map, show
#    import nibabel as nib
#    
#    # Plot the mean image because we have no anatomic data
#    mean_img = image.mean_img(func_filename)
#    weight_img = nib.load('haxby_svc_weights.nii')
#    plot_stat_map(weight_img, mean_img, title='SVM weights')
#    show()
    
    
    
    