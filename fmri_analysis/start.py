# -*- coding: utf-8 -*-
"""
记录一些常用的语句
"""
import numpy as np
label = np.array([1,2,3,4,5,6,7,8,9,0])
np.save('label.npy',label) #保存numpy变量

import os
path = r"D:\data_processing\Python"
os.chdir(path)

import nibabel as nib
path = r"d:\c2s20160713001-193508-00001-00176-1.img"
img = nib.load(path)
img_data = img.get_data()

