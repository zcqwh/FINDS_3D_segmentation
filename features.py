# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 17:48:33 2023

@author: Nana
"""
import os
from skimage.measure import regionprops
import h5py
import numpy as np
import SimpleITK as sitk
import threading
from scipy.spatial import ConvexHull
import skimage.morphology as morphology
from scipy.ndimage import label

properties_save = ['area',
                   'area_convex',
                   'area_filled',
                   'axis_major_length',
                   'axis_minor_length',
                   'bbox',
                   'centroid',
                   'centroid_local',
                   'centroid_weighted',
                   'centroid_weighted_local',
                   #'coords',
                   #'coords_scaled',
                   'equivalent_diameter_area',
                   'euler_number',
                   'extent',
                   'feret_diameter_max',
                   'inertia_tensor',
                   'inertia_tensor_eigvals',
                   'intensity_max',
                   'intensity_mean',
                   'intensity_min',
                   'label',
                   'moments',
                   'moments_central',
                   'moments_normalized',
                   'moments_weighted',
                   'moments_weighted_central',
                   'moments_weighted_normalized',
                   'num_pixels',
                   'solidity'
                   # 'image',
                   # 'image_convex',
                   # 'image_filled',
                   # 'image_intensity',
                   ]

body_properties_save = ['area',
                        'area_convex',
                        'area_filled',
                        'axis_major_length',
                        'axis_minor_length',
                        'bbox',
                        'centroid',
                        'centroid_local',
                        'centroid_weighted',
                        'centroid_weighted_local',
                        #'coords',
                        #'coords_scaled',
                        'equivalent_diameter_area',
                        'euler_number',
                        'extent',
                        # 'feret_diameter_max',
                        'inertia_tensor',
                        'inertia_tensor_eigvals',
                        'intensity_max',
                        'intensity_mean',
                        'intensity_min',
                        'label',
                        'moments',
                        'moments_central',
                        'moments_normalized',
                        'moments_weighted',
                        'moments_weighted_central',
                        'moments_weighted_normalized',
                        'num_pixels',
                        # 'solidity'
                        ]
# 定义文件夹路径
disc_mask_folder = r'Z:\Nana\FINDS_task\data\Prediction\Dataset503_FINDS\3d_fullres_pp'
body_mask_folder = r'Z:\Nana\FINDS_task\data\DATASET\Task503_FINDS\imagesTs_mask'
intensity_folder = r'Z:\Nana\FINDS_task\data\DATASET\Task503_FINDS\imagesTs'
h5_file_path = 'Task503_features_20240329.h5'
# pickle_file = 'region_features.pickle'
pixel_size = (0.008, 0.008, 0.008)  # unit: mm
pixel_volume = pixel_size[0] * pixel_size[1] * pixel_size[2]
pixel_length = pixel_size[0]

# 创建一个字典来按标签存储属性
label_props_dict = {}

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
            single_connected_domain = True
            for label_to_check in [3, 4, 5, 6]:
                mask = disc_image == label_to_check
                _, num_features = label(mask)
                if num_features != 1:
                    single_connected_domain = False
                    break
                    
            if single_connected_domain:
                intensity_image = sitk.ReadImage(os.path.join(intensity_folder, filename))
                intensity_image = sitk.GetArrayFromImage(intensity_image)
    
                # body
                body_mask = sitk.ReadImage(os.path.join(body_mask_folder, filename))
                body_mask = sitk.GetArrayFromImage(body_mask)
                body_regions = regionprops(body_mask, intensity_image, spacing=pixel_size)
                for region in body_regions:
                    if 'body' not in label_props_dict:
                        label_props_dict['body'] = {prop: [] for prop in properties_save}
    
                    for prop in body_properties_save:
                        if prop == 'filename':
                            label_props_dict['body']['filename'].append(filename)
                            
                        elif prop == 'area_convex':
                            points = getattr(region, 'coords')
                            convexhull = ConvexHull(points).volume * pixel_volume
                            label_props_dict['body'][prop].append(convexhull)
                            label_props_dict['body']['solidity'].append(getattr(region, 'area') / convexhull)
                            
                        elif prop == 'feret_diameter_max':
                            skeleton = morphology.skeletonize_3d(body_mask)
                            feret_diameter = regionprops(skeleton)[0].feret_diameter_max*pixel_length
                            label_props_dict['body'][prop].append(feret_diameter)
                            
                        else:
                            try:
                                value = getattr(region, prop)
                                if isinstance(value, np.ndarray):
                                    label_props_dict['body'][prop].append(value)
                                elif isinstance(value, tuple):
                                    label_props_dict['body'][prop].append(list(value))
                                else:
                                    label_props_dict['body'][prop].append(value)
                            except Exception as e:
                                label_props_dict['body'][prop].append(None)
                                print(f"Error processing property '{prop}' for region: {e}")
    
                # 处理disc每个区域的属性
                disc_props = regionprops(disc_image, intensity_image, spacing=pixel_size)
                for i, region in enumerate(disc_props):
                    region_label = i + 1  # 区域标签从1开始
                    if region_label not in label_props_dict:
                        label_props_dict[region_label] = {prop: [] for prop in properties_save}
                        label_props_dict[region_label]['filename'] = []
    
                    for prop in properties_save:
                        if prop == 'filename':
                            label_props_dict[region_label]['filename'].append(filename)
                        else:
                            try:
                                value = getattr(region, prop)
                                if isinstance(value, np.ndarray):
                                    label_props_dict[region_label][prop].append(value)
    
                                elif isinstance(value, tuple):
                                    label_props_dict[region_label][prop].append(list(value))
                                else:
                                    label_props_dict[region_label][prop].append(value)
                            except:
                                label_props_dict[region_label][prop].append(None)
    
            else:
                with print_lock:
                    print(f"Exclude {filename}:{unique_labels}")
                    
        else:
            with print_lock:
                print(f"Exclude {filename}:{unique_labels}")

        # 更新已处理文件的计数器
        with progress_lock:
            processed_files += 1
            progress = (processed_files / total_files) * 100
            print(f"Processed {filename} {processed_files}/{total_files} files. Progress: {progress:.2f}%")


# 定义线程数
num_threads = 64  # 根据需要更改线程数
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
# 打开HDF5文件以写入数据
with h5py.File(h5_file_path, 'w') as h5file:
    h5file.attrs['unit'] = 'mm'
    h5file.attrs['pixel_size'] = pixel_size

    # 遍历字典中的每个标签
    for region_label, props in label_props_dict.items():
        # 为每个标签创建一个组
        group = h5file.create_group(f'Label_{region_label}')

        # 遍历当前标签的每个属性列表
        for prop_name, prop_values in props.items():
            if prop_name in ['coords', 'coords_scaled']:
                pass
            else:
                if isinstance(prop_values, np.ndarray):
                    group.create_dataset(prop_name, data=prop_values)
                elif isinstance(prop_values, (int, float, np.int32, np.float64)):
                    group.create_dataset(prop_name, data=prop_values)
                elif isinstance(prop_values, (tuple, list)):
                    group.create_dataset(prop_name, data=np.array(prop_values).astype('float64'))
                else:
                    group.create_dataset(prop_name, data=str(prop_values))

                # if prop_name == 'feret_diameter_max':
                #     dataset.attrs['description'] = 'Maximum Feret diameter of skeleton'

print("数据已成功保存到HDF5文件。")
