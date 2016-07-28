# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:47:52 2016
使用mask提取感兴趣的脑区，使用SVM分类
@author: FF120
"""
import numpy as np
from nilearn.input_data import NiftiMasker
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score

label_path = r"D:\FF120\workspace\Python\code\fmri\nilearn_learn\haxby2001\subj1\labels.txt"
#1452个标签，对应1452个volum的类别
labels = np.recfromcsv(label_path, delimiter=" ")

target = labels['labels']

# Keep only data corresponding to faces or cats
condition_mask = np.logical_or(labels['labels'] == b'face',
                               labels['labels'] == b'cat')
target = target[condition_mask]
##加载模版，使用模版与被试图像的与  选择某一大脑区域的体素进行分类
mask_filename = r"D:\FF120\workspace\Python\code\fmri\nilearn_learn\haxby2001\subj1\mask4_vt.nii.gz"

nifti_masker = NiftiMasker(mask_img=mask_filename, standardize=True)

func_filename=r"D:\FF120\workspace\Python\code\fmri\nilearn_learn\haxby2001\subj1\bold.nii.gz"
#功能像和模版与，同时转换成2D图像，体素*时间
fmri_masked = nifti_masker.fit_transform(func_filename)
##选择那些需要分类的数据对应的体素，现在fmri_masled是216*577，最后分类的数据是216 volume * 577体素
##每个volume对应的类别标签在labels中
fmri_masked = fmri_masked[condition_mask]

#############################################
#特征提取，特征选择，特征转换都做完，准备好了机器学习所需要的数据，下一步使用SVM分类
###############
#指定使用线性核
svc = SVC(kernel='linear')
#    ##训练模型
#    svc.fit(fmri_masked, target)
#    ##预测分类
#    prediction = svc.predict(fmri_masked)
#    ##统计准确率,可以看到，预测的准确率为1，因为这里特征较少，训练和测试使用的都是相同的数据，
#    ##实际使用中不能这样，要使用交叉验证
#    right_rate = np.sum(prediction == target) / len(target)


#使用交叉验证 训练和测试
cv = KFold(n=len(fmri_masked), n_folds=5)
##记录结果
cv_scores = []

for train, test in cv:
    svc.fit(fmri_masked[train], target[train])
    prediction = svc.predict(fmri_masked[test])
    cv_scores.append(np.sum(prediction == target[test]) / float(np.size(target[test])))

#预测准确率
print(cv_scores)
#平均准确率
mean_score = np.mean(cv_scores)
print(cv_scores)

#    #不适用上面的for循环自己写，也可以使用scikit-learn实现好的一个函数实现准确率的计算
#    # n_jobs=-1 使用所有的计算核心计算
#    cv_scores_right_rate = cross_val_score(svc, fmri_masked, target, cv=cv) 
#    print "right rate:"+str(cv_scores_right_rate)

#使用其他方法计算得分
 
#计算某个分类的changce 水平
from sklearn.dummy import DummyClassifier
null_cv_scores = cross_val_score(DummyClassifier(), fmri_masked, target, cv=cv)
print "changce level:"+str(null_cv_scores)

#    #置换检验
#    from sklearn.cross_validation import permutation_test_score
#    null_cv_scores_permutation = permutation_test_score(svc, fmri_masked, target, cv=cv) 
#    print "使用置换检验的changce level:"+str(null_cv_scores_permutation)

##解码结果的可视化
##暂时的解决方案###########
#svc.fit(fmri_masked, target)
##svc.coef_ 训练模型时给特征分配的权重
coef_ = svc.coef_

coef_img = nifti_masker.inverse_transform(coef_)
#显示图像
from nilearn.image import mean_img
from nilearn.plotting import plot_roi, plot_stat_map, show

mean_epi = mean_img(func_filename)
plot_stat_map(coef_img, mean_epi, title="SVM weights", display_mode="yx")
#Plot also the mask that was computed by the NiftiMasker
plot_roi(nifti_masker.mask_img_, mean_epi, title="Mask", display_mode="yx")
show()