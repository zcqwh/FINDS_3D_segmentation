import numpy as np
import os
import csv
from concurrent.futures import ProcessPoolExecutor
import h5py
from skimage import io
from skimage.filters import threshold_otsu
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter, binary_fill_holes, label


"""
用于预处理三维图像数据的Python脚本
对图像进行了threshold_otsu,去除背景。 threshold_range：（100，1000）。
保留最大region
"""

def apply_otsu(image, lower_limit, upper_limit):
    subset_image = image[(image > lower_limit) & (image < upper_limit)]
    if subset_image.size > 0:
        threshold = threshold_otsu(subset_image)
    else:
        threshold = lower_limit

    image_copy = image.copy()
    image_copy[image_copy < threshold] = 0

    return image_copy

def process_image(image_path, sigma=2):
    """
        Process an image from an HDF5 file.

        Parameters:
        image_path (str): The path to the HDF5 file containing the image data.
        sigma (float): The standard deviation for the Gaussian filter (default is 2).

        Returns:
        numpy.ndarray: The processed image.
        numpy.ndarray: max_region_mask.
    """

    with h5py.File(image_path, 'r') as file:
        image = np.array(file['dataset_1'])
        image = apply_otsu(image, 100, 1000)

    smoothed_image = gaussian_filter(image, sigma=sigma)
    threshold_image = np.where(smoothed_image > 0, 1, 0)

    labeled_image, _ = label(threshold_image)

    region_sizes = np.bincount(labeled_image.ravel())
    max_region_label = region_sizes[1:].argmax() + 1

    max_region_mask = (labeled_image == max_region_label)
    max_region_mask = binary_fill_holes(max_region_mask)
    image = image * max_region_mask

    return image, max_region_mask

def process_single_item(task_details):
    index, row, save_paths = task_details
    image_path, label_path = row
    save_label_path, save_train_path, save_test_path, save_train_mask_path, save_test_mask_path = save_paths

    # Process the label if it exists
    if label_path != "None":
        label_file_path = os.path.join(save_label_path, f"FINDS_{index:04d}.nii.gz")
        if not os.path.exists(label_file_path):
            if label_path.endswith('.h5'):
                with h5py.File(label_path, 'r') as file:
                    label_img = np.array(file['dataset_1'])
            elif label_path.endswith('.tif'):
                label_img = io.imread(label_path)
                label_img = np.transpose(label_img, (2, 1, 0))

            sitk_label = sitk.GetImageFromArray(label_img)
            sitk.WriteImage(sitk_label, label_file_path)

    # Process the image (assuming it's always present)
    processed_image, mask = process_image(image_path)  # Your existing function to process the image
    image_save_path = os.path.join(save_train_path if label_path != "None" else save_test_path, f"FINDS_{index:04d}.nii.gz")
    mask_save_path = os.path.join(save_train_mask_path if label_path != "None" else save_test_mask_path, f"FINDS_{index:04d}.nii.gz")

    # Save the processed image and mask
    sitk_image = sitk.GetImageFromArray(processed_image.astype(np.float32))
    sitk.WriteImage(sitk_image, image_save_path)

    sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    sitk.WriteImage(sitk_mask, mask_save_path)

    print(f'Processed dataset {index}: {os.path.basename(image_path)}  Label: {label_path}')


# Main function to orchestrate the multiprocessing workflow
def main():
    num = 510
    data_sheet = r"data_label_pairs_4hittest.csv"

    target_folder = f"Z:\\Nana\\FINDS_task\\data\\DATASET\\Task{num}_4hittest"
    save_label_path = os.path.join(target_folder, "labelsTr")
    save_train_path = os.path.join(target_folder, "imagesTr")
    save_test_path = os.path.join(target_folder, "imagesTs")
    save_train_mask_path = os.path.join(target_folder, "imagesTr_mask")
    save_test_mask_path = os.path.join(target_folder, "imagesTs_mask")

    # 检查每个路径是否存在，如果不存在则创建
    for path in [save_label_path, save_train_path, save_test_path, save_train_mask_path, save_test_mask_path]:
        os.makedirs(path, exist_ok=True)

    # Define save_paths correctly before using it
    save_paths = [save_label_path, save_train_path, save_test_path, save_train_mask_path, save_test_mask_path]

    tasks = []
    with open(data_sheet, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for index, row in enumerate(reader):
            tasks.append((index, row, save_paths))

    # Specify the number of processes you want to use
    num_processes = 16

    # # # Using ProcessPoolExecutor to process each image in parallel with a specified number of processes
    # with ProcessPoolExecutor(max_workers=num_processes) as executor:
    #     executor.map(process_single_item, tasks)
    for task in tasks:
        process_single_item(task)




if __name__ == "__main__":
    main()



# num = 503
# save_image = True
# data_sheet = r"data_label_pairs_240205.csv"
#
# tagert_folder = f"Z:\\Nana\\FINDS_task\\data\\DATASET\\Task{num}_FINDS"
# save_label_path = "labelsTr"
# save_train_path = "imagesTr"
# save_test_path = "imagesTs"
# save_train_mask_path = "imagesTr_mask"
# save_test_mask_path = "imagesTs_mask"
#
# # 检查每个路径是否存在，如果不存在则创建
# for path in [save_label_path, save_train_path, save_test_path, save_train_mask_path, save_test_mask_path]:
#     if not os.path.exists(path):
#         os.makedirs(path)
#         print(f"Created directory: {path}")
#     else:
#         print(f"Directory already exists: {path}")
#
# # 打开 CSV 文件并按行遍历
# with open(data_sheet, 'r', encoding='utf-8') as file:
#     reader = csv.reader(file)
#     next(reader)  # 跳过标题行
#     for index, row in enumerate(reader):
#         image_path = row[0]
#         label_path = row[1]
#
#         if label_path != "None":
#             label_file_path = os.path.join(save_label_path, f"FINDS_{index:04d}.nii.gz")
#             if not os.path.exists(label_file_path):
#                 label = io.imread(label_path)
#                 label = np.transpose(label, (2, 1, 0))
#
#                 sitk_label = sitk.GetImageFromArray(label)
#                 sitk.WriteImage(sitk_label, os.path.join(save_label_path, f"FINDS_{index:04d}.nii.gz"))
#                 print(f"Saved label {label_file_path}")
#
#                 if save_image:
#                     with h5py.File(image_path, 'r') as file:
#                         image = np.array(file['dataset_1'])
#                         image = apply_otsu(image, 100, 1000)
#
#                     processed_image, mask = process_image(image)
#
#                     sitk_image = sitk.GetImageFromArray(image)
#                     sitk.WriteImage(sitk_image, os.path.join(save_train_path, f"FINDS_{index:04d}.nii.gz"))
#
#                     sitk_image = sitk.GetImageFromArray(mask.astype(np.uint8))
#                     sitk.WriteImage(sitk_image, os.path.join(save_train_mask_path, f"FINDS_{index:04d}.nii.gz"))
#
#                     print(f"Processed training dataset {index + 1}: {os.path.basename(label_path)}")
#             else:
#                 print(f'{label_file_path} already exist.')
#
#         else:
#             if save_image:
#                 with h5py.File(image_path, 'r') as file:
#                     image = np.array(file['dataset_1'])
#                     image = apply_otsu(image, 100, 1000)
#
#                 processed_image, mask = process_image(image)
#
#                 sitk_image = sitk.GetImageFromArray(image)
#                 sitk.WriteImage(sitk_image, os.path.join(save_test_path, f"FINDS_{index:04d}.nii.gz"))
#
#                 sitk_image = sitk.GetImageFromArray(mask.astype(np.uint8))
#                 sitk.WriteImage(sitk_image, os.path.join(save_test_mask_path, f"FINDS_{index:04d}.nii.gz"))
#
#                 print(f"Processed test dataset {index + 1}: {os.path.basename(label_path)}")
#
# print("Processing complete.")
