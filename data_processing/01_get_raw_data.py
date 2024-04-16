import os
import glob
import csv

def normalize_and_compare(file1, file2):
    substrings_to_remove = []#[".dcimg", ".labels", ".tif", "_EN", "_CH", ".h5", "_", " ", "-"]

    def remove_substrings(filename):
        for substring in substrings_to_remove:
            filename = filename.replace(substring, "")
        return filename

    normalized_file1 = remove_substrings(file1)
    normalized_file2 = remove_substrings(file2)

    return normalized_file1 == normalized_file2

def write_to_csv(data_label_pairs, csv_file_path):
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Data Path', 'Label Path'])
        writer.writerows(data_label_pairs)


# src_folder = r'Y:\segmentation2'
# csv_file_path = os.path.join('data_label_pairs_240214.csv')
#
# img_quality = 'good'
# label_paths = glob.glob(os.path.join(src_folder, '**', img_quality, '*labels*.tif'), recursive=True)
# file_paths = glob.glob(os.path.join(src_folder, '**', img_quality, '*.h5'), recursive=True)


src_folder = r'X:\ptc.mcherry\20mWpower_affined\Data_for_unet3D\to_test_4_hits'
csv_file_path = os.path.join('data_label_pairs_4hittest.csv')
file_paths = glob.glob(os.path.join(src_folder, '*.h5'), recursive=True)
label_paths = glob.glob(os.path.join(src_folder, 'mask2', '*.h5'), recursive=True)

data_label_pairs = []
matched_count = 0  # 初始化匹配计数
unmatched_count = 0  # 初始化不匹配计数
matched_labels = set()  # 用于存储已匹配的标签文件

for data_file in file_paths:
    data_basename = os.path.basename(data_file)
    matched_label = None

    for label_file in label_paths:
        label_basename = os.path.basename(label_file)
        if normalize_and_compare(data_basename, label_basename):
            # print(f"data_basename: {data_basename}")
            # print(f"label_basename:{label_basename}")
            matched_label = label_file
            matched_labels.add(label_file)
            matched_count += 1  # 增加匹配计数
            break

    if not matched_label:
        unmatched_count += 1  # 增加不匹配计数

    data_label_pairs.append([data_file, matched_label or 'None'])

write_to_csv(data_label_pairs, csv_file_path)

print(f"Training dataset: {matched_count}")
print(f"Test dataset: {unmatched_count}")

# 找出未匹配的标签文件
unmatched_labels = set(label_paths) - matched_labels
if unmatched_labels:
    print("Unmatched label files:")
    for label in unmatched_labels:
        print(label)
else:
    print("All label files are matched.")
