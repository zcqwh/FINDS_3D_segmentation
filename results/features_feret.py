# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 14:56:44 2024

@author: Nana
"""
import os
from skimage.measure import regionprops, label
import h5py
import numpy as np
import SimpleITK as sitk
import threading
from scipy.spatial import ConvexHull
import time
import pickle
import skimage.morphology as morphology
import skimage.io as io
import SimpleITK as sitk
import scipy.ndimage as ndi
from scipy.ndimage import convolve, generate_binary_structure
from scipy.ndimage import generate_binary_structure, binary_dilation
import gc


def calculate_skeleton_length(skeleton):
    # 定义一个3D结构元素，表示体素的26邻域
    struct = generate_binary_structure(3, 3)

    # 使用二值膨胀找到所有骨架体素的邻居
    dilated_skeleton = binary_dilation(skeleton, structure=struct)

    # 计算骨架和其膨胀版本的交集，得到边缘体素
    edge_voxels = dilated_skeleton & ~skeleton

    # 骨架长度可以近似为骨架体素数量加上边缘体素数量的一半
    # 这个近似假设了每个体素至少与一个边缘体素相邻
    length = np.sum(skeleton) + 0.5 * np.sum(edge_voxels)

    return length


properties_save = ['feret_diameter_max']
# 定义文件夹路径
disc_mask_folder = r'Z:\Nana\FINDS_task\data\Prediction\Dataset503_FINDS\3d_fullres_pp'
body_mask_folder = r'Z:\Nana\FINDS_task\data\DATASET\Task503_FINDS\imagesTs_mask'
intensity_folder = r'Z:\Nana\FINDS_task\data\DATASET\Task503_FINDS\imagesTs'
h5_file_path = 'region_features_cascade_fullres_new_data_240212.h5'
# pickle_file = 'region_features.pickle'
pixel_size = (0.008, 0.008, 0.008)  # unit: mm
pixel_volume = pixel_size[0] * pixel_size[1] * pixel_size[2]

# 创建一个字典来按标签存储属性
feret_diameters = []

# 初始化锁
print_lock = threading.Lock()
progress_lock = threading.Lock()

# 共享变量，用于跟踪处理的文件数量
processed_files = 0
total_files = 0


def process_file(filenames):
    global processed_files
    for filename in filenames:
        # 加载NII文件
        disc_image = sitk.ReadImage(os.path.join(disc_mask_folder, filename))
        disc_image = sitk.GetArrayFromImage(disc_image)
        unique_labels = np.unique(disc_image.astype('uint8'))
        expected_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7])

        # 检查unique_labels是否包含expected_labels中的所有元素
        if np.array_equal(np.sort(unique_labels), np.sort(expected_labels)):
            # intensity_image = sitk.ReadImage(os.path.join(intensity_folder, filename))
            # intensity_image = sitk.GetArrayFromImage(intensity_image)

            # body
            body_mask = sitk.ReadImage(os.path.join(body_mask_folder, filename))
            body_mask = sitk.GetArrayFromImage(body_mask)

            # 使用骨架提取算法来获取骨架
            skeleton = morphology.skeletonize_3d(body_mask)
            feret_diameter = regionprops(skeleton)[0].feret_diameter_max * 0.008
            feret_diameters.append(feret_diameter)

        # 更新已处理文件的计数器
        with progress_lock:
            processed_files += 1
            progress = (processed_files / total_files) * 100
            print(f"Processed {filename} {processed_files}/{total_files} files. Progress: {progress:.2f}%")

        gc.collect()  # 强制进行垃圾收集


# 定义线程数
num_threads = 1  # 根据需要更改线程数
# 获取mask文件夹中的所有文件
file_list = [filename for filename in os.listdir(disc_mask_folder) if filename.endswith('.nii.gz')]
total_files = len(file_list)

# 创建并启动多个线程来处理文件
threads = []
for i in range(num_threads):
    thread = threading.Thread(target=process_file, args=(file_list[i::num_threads],))
    thread.start()
    threads.append(thread)
# 等待所有线程完成
for thread in threads:
    thread.join()

# %%
with h5py.File(h5_file_path, 'a') as h5file:  # 使用'a'模式以便读写且不删除原有内容
    # 检查'label_body'组是否存在，不存在则创建
    if 'Label_body' not in h5file:
        label_body_group = h5file.create_group('Label_body')
    else:
        label_body_group = h5file['Label_body']

    # 检查'feret_diameter_max'数据集是否存在，存在则删除
    if 'feret_diameter_max' in label_body_group:
        del label_body_group['feret_diameter_max']

    # 创建并写入新的'feret_diameter_max'数据集
    label_body_group.create_dataset('feret_diameter_max', data=np.array(feret_diameters))

    # 可选：给数据集添加属性
    label_body_group['feret_diameter_max'].attrs['description'] = 'Maximum Feret diameter of body skeleton'

# =============================================================================
# 
# # %%
# # 打开HDF5文件以写入数据
# with h5py.File(h5_file_path, 'w') as h5file:
#     h5file.attrs['unit'] = 'mm'
#     h5file.attrs['pixel_size'] = pixel_size
# 
#     # 遍历字典中的每个标签
#     for region_label, props in label_props_dict.items():
#         # 为每个标签创建一个组
#         group = h5file.create_group(f'Label_{region_label}')
# 
#         # 遍历当前标签的每个属性列表
#         for prop_name, prop_values in props.items():
#             if prop_name in ['coords','coords_scaled']:
#                 pass
#             else:
#                 if isinstance(prop_values, np.ndarray):
#                        group.create_dataset(prop_name, data=prop_values)
#                 elif isinstance(prop_values, (int, float, np.int32, np.float64)):
#                     group.create_dataset(prop_name, data=prop_values)
#                 elif isinstance(prop_values, (tuple, list)):
#                     group.create_dataset(prop_name, data=np.array(prop_values).astype('float64'))
#                 else:
#                     group.create_dataset(prop_name, data=str(prop_values))
# 
#                 
# print("数据已成功保存到HDF5文件。")
# =============================================================================
