####################################
##核磁数据处理常用的操作
######################################
import nibabel as nib  #引入nibabel 读取数据
import numpy as np
import os
#读取一个img  hdr对；
img = nib.load("fauditory001-0008-00001-000001-01.img")
#读取一个文件夹下所有的img hdr对

# img  <nibabel.nifti1.Nifti1Pair at 0x194d2f98>
# Nifti1Pair类型的img对象包含了图像的所有信息，包括头信息，与标准空间的映射信息，图像矩阵的像素信息
# img.shape
# img.get_data_dtype()
# img.affine() 仿射变换矩阵
# img.dataobj()  数组代理
img_data = img.get_data()

#设置np.array的显示方式
np.set_printoptions(precision=2, suppress=True)

#从img获取头信息
img_hdr = img.header();
#img_hdr.get_data_dtype()
#img_hdr.get_data_shape()
#img_hdr.get_zooms() 得到体素的大小，以毫米为单位

#保存图像
nib.save(img,"fi.img") #会在磁盘保存一个fi.img fi.hdr 对

#查看图像包含的文件
list(img.file_map)
Out[40]: ['header', 'image']

img.file_map['image'].filename
Out[41]: 'fi.img'

img.file_map['header'].filename

Out[42]: 'fi.hdr'

#显示某一层的图像
import matplotlib.pyplot as plt
>>> def show_slices(slices):
...    """ Function to display row of image slices """
...    fig, axes = plt.subplots(1, len(slices))
...    for i, slice in enumerate(slices):
...        axes[i].imshow(slice.T, cmap="gray", origin="lower")
>>>

#遍历序列和下标  enumerate
for i,value in enumerate(some_array)

#显示图像
slice0 = img_data[2,:,:]

slice1 = img_data[:,2,:]

slice2 = img_data[:,:,3]

  ([slice0,slice2,slice2])

#或者使用nilearn提供的方法
from nilearn import plotting
cut_coords = [2,2,2]
plotting.plot_anat(img,cut_coords=cut_coords,title='标题')

#直接将某个矩阵保存成图像
from scipy.misc import imsave
imsave("aa1.png",slice1)

#在nilearn中，一个3-D图像的list可以当作一个4-D图像

