# -*- coding: utf-8 -*-
from nilearn import datasets
from nilearn.input_data import NiftiMasker
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import LeaveOneLabelOut
from sklearn.cross_validation import permutation_test_score
from sklearn.dummy import DummyClassifier
import numpy as np
###############################################
## 使用感兴趣区域选择的体素，使得体素数目较少
###############################################
haxby_dataset = datasets.fetch_haxby()

labels = np.recfromcsv(haxby_dataset.session_target[0], delimiter=" ")

target = labels['labels']
# Keep only data corresponding to faces or cats
condition_mask = np.logical_or(labels['labels'] == b'face',
                               labels['labels'] == b'cat')
#准备好了标签Y                             
target = target[condition_mask]
#模版文件的路径
mask_filename = haxby_dataset.mask_vt[0]
#加载模版并标准化
nifti_masker = NiftiMasker(mask_img=mask_filename, standardize=True)

func_filename = haxby_dataset.func[0]

fmri_masked = nifti_masker.fit_transform(func_filename)
#准备好了特征
fmri_masked = fmri_masked[condition_mask]

#使用SVM分类和预测
svc = SVC(kernel='linear')
#训练模型
#svc.fit(fmri_masked, target)
#预测
#prediction = svc.predict(fmri_masked)

#使用训练的数据预测结果很可能达到100%；
#实际应用的时候通常使用交叉验证的方式


cv = KFold(n=len(fmri_masked), n_folds=5)

cv_scores = []

for train, test in cv:
    svc.fit(fmri_masked[train], target[train])
    prediction = svc.predict(fmri_masked[test])
    cv_scores.append( np.sum(prediction == target[test]) / float(np.size(target[test])) )

cv_scores = cross_val_score(svc, fmri_masked, target, cv=cv)  
#下面的可以加快计算速度
#cv_scores = cross_val_score(svc, fmri_masked, target, cv=cv, n_jobs=-1, verbose=10) 

session_label = labels['chunks']  
session_label = session_label[condition_mask] 
cv = LeaveOneLabelOut(labels=session_label)  
cv_scores_one = cross_val_score(svc, fmri_masked, target, cv=cv) 
#使用F1评分
#cv_scores = cross_val_score(svc, fmri_masked, target, cv=cv,  scoring='f1')  
#计算平均分类准确率
classification_accuracy = np.mean(cv_scores)

classification_accuracy_one = np.mean(cv_scores_one)
#计算随机分类器的交叉验证得分
null_cv_scores = cross_val_score(DummyClassifier(), fmri_masked, target, cv=cv)  
#置换检验
null_cv_scores_2 = permutation_test_score(svc, fmri_masked, target, cv=cv)  

# Retrieve the SVC discriminating weights
coef_ = svc.coef_

# Reverse masking thanks to the Nifti Masker
coef_img = nifti_masker.inverse_transform(coef_)

# Save the coefficients as a Nifti image
coef_img.to_filename('haxby_svc_weights.nii')


from nilearn import image
from nilearn.plotting import plot_stat_map, show
import nibabel as nib

# Plot the mean image because we have no anatomic data
mean_img = image.mean_img(func_filename)
weight_img = nib.load('haxby_svc_weights.nii')
plot_stat_map(weight_img, mean_img, title='SVM weights')
show()
    
