####################################
##�˴����ݴ����õĲ���
######################################
import nibabel as nib  #����nibabel ��ȡ����
import numpy as np
import os
#��ȡһ��img  hdr�ԣ�
img = nib.load("fauditory001-0008-00001-000001-01.img")
#��ȡһ���ļ��������е�img hdr��

# img  <nibabel.nifti1.Nifti1Pair at 0x194d2f98>
# Nifti1Pair���͵�img���������ͼ���������Ϣ������ͷ��Ϣ�����׼�ռ��ӳ����Ϣ��ͼ������������Ϣ
# img.shape
# img.get_data_dtype()
# img.affine() ����任����
# img.dataobj()  �������
img_data = img.get_data()

#����np.array����ʾ��ʽ
np.set_printoptions(precision=2, suppress=True)

#��img��ȡͷ��Ϣ
img_hdr = img.header();
#img_hdr.get_data_dtype()
#img_hdr.get_data_shape()
#img_hdr.get_zooms() �õ����صĴ�С���Ժ���Ϊ��λ

#����ͼ��
nib.save(img,"fi.img") #���ڴ��̱���һ��fi.img fi.hdr ��

#�鿴ͼ��������ļ�
list(img.file_map)
Out[40]: ['header', 'image']

img.file_map['image'].filename
Out[41]: 'fi.img'

img.file_map['header'].filename

Out[42]: 'fi.hdr'

#��ʾĳһ���ͼ��
import matplotlib.pyplot as plt
>>> def show_slices(slices):
...    """ Function to display row of image slices """
...    fig, axes = plt.subplots(1, len(slices))
...    for i, slice in enumerate(slices):
...        axes[i].imshow(slice.T, cmap="gray", origin="lower")
>>>

#�������к��±�  enumerate
for i,value in enumerate(some_array)

#��ʾͼ��
slice0 = img_data[2,:,:]

slice1 = img_data[:,2,:]

slice2 = img_data[:,:,3]

  ([slice0,slice2,slice2])

#����ʹ��nilearn�ṩ�ķ���
from nilearn import plotting
cut_coords = [2,2,2]
plotting.plot_anat(img,cut_coords=cut_coords,title='����')

#ֱ�ӽ�ĳ�����󱣴��ͼ��
from scipy.misc import imsave
imsave("aa1.png",slice1)

#��nilearn�У�һ��3-Dͼ���list���Ե���һ��4-Dͼ��

