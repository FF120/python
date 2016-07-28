# -*- coding: utf-8 -*-
from nilearn import datasets
from nilearn.input_data import NiftiMasker
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import LeaveOneLabelOut
from sklearn.cross_validation import permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
import numpy as np
###############################################
## 使用统计的方法选择体素-特征选择
###############################################
haxby_dataset = datasets.fetch_haxby()

labels = np.recfromcsv(haxby_dataset.session_target[0], delimiter=" ")

target = labels['labels']
# Keep only data corresponding to faces or cats
condition_mask = np.logical_or(labels['labels'] == b'face',
                               labels['labels'] == b'cat')
#准备好了标签Y                             
target = target[condition_mask]
###########################################################
#选择500维特征
feature_selection = SelectKBest(f_classif, k=500)

#############################################################################
# Prepare the fMRI data
mask_filename = haxby_dataset.mask
# For decoding, standardizing is often very important
nifti_masker = NiftiMasker(mask_img=mask_filename, sessions=session,
                           smoothing_fwhm=4, standardize=True,
                           memory="nilearn_cache", memory_level=1)
func_filename = haxby_dataset.func[0]
X = nifti_masker.fit_transform(func_filename)
# Apply our condition_mask
X = X[condition_mask]


#使用SVM分类和预测
svc = SVC(kernel='linear')

anova_svc = Pipeline([('anova', feature_selection), ('svc', svc)])

# Fit the decoder and predict


anova_svc.fit(X, y)
y_pred = anova_svc.predict(X)


