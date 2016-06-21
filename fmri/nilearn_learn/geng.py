# -*- coding: utf-8 -*-
"""
Created on Tue May 31 18:24:39 2016
使用自己的数据和自己制作的模版解码
@author: FF120
"""
import numpy as np
from nilearn.input_data import NiftiMasker
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
import scipy.io as sio 

label_path = r"D:\FF120\workspace\Python\code\fmri\nilearn_learn\geng\label.mat"

labels = sio.loadmat(label_path) 

target = np.array(labels['label'])
##加载模版，使用模版与被试图像的与  选择某一大脑区域的体素进行分类
#mask_filename = r"D:\FF120\workspace\Python\code\fmri\nilearn_learn\geng\fSTG.mn.nii.gz"
#mask_filename = r"D:\FF120\workspace\Python\code\fmri\nilearn_learn\geng\Resliced_STG.mn.nii"
mask_filename = r"D:\FF120\workspace\Python\code\fmri\nilearn_learn\geng\fNiftiPairs_Resliced_STG.mn.nii"
nifti_masker = NiftiMasker(mask_img=mask_filename, standardize=False)

func_filename=r"D:\FF120\workspace\Python\code\fmri\nilearn_learn\geng\20160306_151338ep2dbold6sstis008a001.nii.gz"
#功能像和模版与，同时转换成2D图像，体素*时间
fmri_masked = nifti_masker.fit_transform(func_filename)


one_volume = target[0:len(target)/2];
one_volume = one_volume.ravel()
#去掉前18s的空闲时间
one_fmri_data = fmri_masked[9:]


svc = SVC(kernel='linear')

#    svc.fit(one_fmri_data, one_volume)
#    ##预测分类
#    prediction = svc.predict(one_fmri_data)
#    ##统计准确率,可以看到，预测的准确率为1，因为这里特征较少，训练和测试使用的都是相同的数据，
#    ##实际使用中不能这样，要使用交叉验证
#    right_rate = np.sum(prediction == one_volume) / len(one_volume)
#    
#    print "准确率："+str(right_rate)

cv = KFold(n=len(one_fmri_data), n_folds=2)
cv_scores = []

for train, test in cv:
    svc.fit(one_fmri_data[train], one_volume[train])
    prediction = svc.predict(one_fmri_data[test])
    cv_scores.append(np.sum(prediction == one_volume[test]) / float(np.size(one_volume[test])))

#cv_scores_right_rate = cross_val_score(svc, one_fmri_data, one_volume, cv=cv) 
print "right rate:"+str(cv_scores)
#平均准确率
mean_score = np.mean(cv_scores)
print "mean right rate:" + str(mean_score)
##计算changce level
from sklearn.dummy import DummyClassifier
null_cv_scores = cross_val_score(DummyClassifier(), one_fmri_data, one_volume, cv=cv)
print "changce level:"+str(null_cv_scores)
print "mean change level:" + str(np.mean(null_cv_scores))



##svc.coef_ 训练模型时给特征分配的权重
coef_ = svc.coef_

coef_img = nifti_masker.inverse_transform(coef_)
#显示图像---有些问题
#from nilearn.image import mean_img
#from nilearn.plotting import plot_roi, plot_stat_map, show
#
#mean_epi = mean_img(func_filename)
#plot_stat_map(coef_img, mean_epi, title="SVM weights", display_mode="yx")
##Plot also the mask that was computed by the NiftiMasker
#plot_roi(nifti_masker.mask_img_, mean_epi, title="Mask", display_mode="yx")
#show()



