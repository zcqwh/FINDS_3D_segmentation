# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:50:49 2024

@author: Nana
"""
import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
from skimage.measure import label
from scipy.ndimage import binary_fill_holes
import random

def get_body_mask(input_image, sigma=5, fill_holes=True):
    """
    提取3D图像中最大连通域的掩膜。

    参数:
    input_image -- 3D二值图像(numpy数组)。
    fill_holes -- 是否填充掩膜中的空洞（默认为False）。

    返回:
    3D掩膜图像，其中最大连通域为1，其他区域为0。
    """
    # Apply Gaussian filter if sigma is provided
    smoothed_image = gaussian_filter(input_image, sigma=sigma) if sigma else input_image

    threshold_image = np.where(smoothed_image > 0, 1, 0)
    
    # 标记图像中的连通域
    labeled_image = label(threshold_image)

    # 计算每个连通域的大小
    region_sizes = np.bincount(labeled_image.ravel())

    # 找到最大连通域的标签（忽略背景标签0）
    max_region_label = region_sizes[1:].argmax() + 1

    # 创建最大连通域的掩膜
    max_region_mask = (labeled_image == max_region_label)

    # 可选：填充掩膜中的空洞
    if fill_holes:
        max_region_mask = binary_fill_holes(max_region_mask)

    return max_region_mask.astype('uint8')

image_folder = r"Z:\Nana\TransUnet_For_FINDS\data\DATASET\Task500_FINDS\imagesTs"
mask_folder = r"Z:\Nana\TransUnet_For_FINDS\data\DATASET\Task500_FINDS\imagesTs_mask"

files = [file for file in os.listdir(image_folder) if file.endswith('.nii.gz')]

# Randomly select 5 files
selected_files = random.sample(files, 1) 
for filename in files:
    print(f'Processing {filename}')
    image = sitk.ReadImage(os.path.join(image_folder, filename))
    image = sitk.GetArrayFromImage(image)
    
    mask = get_body_mask(image,sigma=2)
    sitk_image = sitk.GetImageFromArray(mask)
    sitk.WriteImage(sitk_image, os.path.join(mask_folder, f"{filename}.nii.gz"))



