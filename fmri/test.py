# -*- coding: utf-8 -*-
"""
Created on Wed Jun 01 09:36:40 2016

@author: FF120
"""
import nibabel as nib  #引入nibabel 读取数据
import numpy as np
import os

mask_filename = r"D:\FF120\workspace\Python\code\fmri\nilearn_learn\geng\fSTG.mn.nii.gz"
img = nib.load(mask_filename)