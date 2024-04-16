import json
import pandas as pd
import os



folder_preds = r'Z:\Nana\FINDS_task\data\DATASET\nnUNet\nnUNet_results\Dataset503_FINDS\nnUNetTrainer__nnUNetPlans__3d_fullres'

for folder in os.listdir(folder_preds):
    # 构造每个条目的完整路径
    folder_pred = os.path.join(folder_preds, folder)
    if os.path.isdir(folder_pred):
        for root, dirs, files in os.walk(folder_pred):
            if 'summary.json' in files:
                file_path = os.path.join(root, 'summary.json')


        output = os.path.join(folder_pred, f'metrics.xlsx')

        # Read the JSON data from the file
        with open(file_path, 'r') as file:
            json_data = json.load(file)

        # Convert the JSON data to a pandas DataFrame
        df = pd.json_normalize(json_data)

        # Define the metrics and rows
        mean_metrics = ['Dice', 'IoU', 'TP', 'TN', 'FN', 'FP']
        rows = ['foreground_mean', 'mean.1', 'mean.2', 'mean.3', 'mean.4', 'mean.5', 'mean.6', 'mean.7']

        # Create a new DataFrame for visualization
        visualized_data = pd.DataFrame()

        for row in rows:
            row_data = []
            for metric in mean_metrics:
                column_name = f"{row}.{metric}" if row != 'foreground_mean' else f"foreground_mean.{metric}"
                row_data.append(df[column_name].iloc[0] if column_name in df.columns else None)
            visualized_data.loc[row, mean_metrics] = row_data

        # Update the index to the desired format
        visualized_data.index = ['foreground', '1', '2', '3', '4', '5', '6', '7']

        # Calculate Recall and Precision
        visualized_data['Recall'] = visualized_data['TP'] / (visualized_data['TP'] + visualized_data['FN'])
        visualized_data['Precision'] = visualized_data['TP'] / (visualized_data['TP'] + visualized_data['FP'])

        # Reorder columns to insert Recall and Precision between IoU and TP
        columns_reordered = ['Dice', 'IoU', 'Recall', 'Precision', 'TP', 'TN', 'FN', 'FP']
        visualized_data = visualized_data[columns_reordered]

        # Save the visualized data to an Excel file
        visualized_data.to_excel(output, index=True)

        print(f"Data has been saved to {output}")

# %%
import json
import pandas as pd
import os


# 加载JSON文件
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


# 转换数据并保存为Excel
def convert_and_save_excel(json_data, output_excel_path):
    metric_per_case = json_data['metric_per_case']

    # 初始化列表来收集所有行的数据
    rows_list = []

    # 初始化一个字典来收集每一类的dice值，用于计算全局平均
    class_dice_values = {str(i): [] for i in range(1, 8)}

    for case in metric_per_case:
        prediction_file = case['prediction_file'].split('\\')[-1]  # 从文件路径提取案例名称
        metrics = case['metrics']
        row = {'case': prediction_file}  # 初始化行数据
        case_dice_values = []  # 存储当前案例的所有dice值
        for i in range(1, 8):  # 类别从1到7
            dice_value = metrics.get(str(i), {}).get('Dice', 0)
            row[str(i)] = dice_value
            case_dice_values.append(dice_value)
            class_dice_values[str(i)].append(dice_value)  # 收集每一类的dice值
        # 计算并添加当前案例的平均Dice值
        row['average'] = sum(case_dice_values) / len(case_dice_values) if case_dice_values else 0
        rows_list.append(row)

    # 计算并添加每一类的全局平均dice值
    average_row = {'case': 'Average'}
    all_dice_values = []  # 存储所有dice值，用于计算总平均
    for class_id, values in class_dice_values.items():
        class_avg = sum(values) / len(values) if values else 0
        average_row[class_id] = class_avg
        all_dice_values.extend(values)  # 收集所有dice值
    # 计算总平均Dice值并添加到平均行
    average_row['average'] = sum(all_dice_values) / len(all_dice_values) if all_dice_values else 0
    rows_list.append(average_row)

    # 使用pd.DataFrame直接从列表创建DataFrame
    formatted_df = pd.DataFrame(rows_list)

    # 保存到Excel文件
    formatted_df.to_excel(output_excel_path, index=False, float_format="%.6f")


json_folder = r"Z:\Nana\FINDS_task\data\DATASET\nnUNet\nnUNet_results\Dataset503_FINDS\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_4\validation"
file_path = os.path.join(json_folder, 'summary.json')  # JSON文件路径
output_excel_path = os.path.join(json_folder, 'case_metrics.xlsx')  # 输出Excel文件路径

json_data = load_json(file_path)
convert_and_save_excel(json_data, output_excel_path)
print(f"Saved {output_excel_path}")

# # main
# json_folder = r'Z:\Nana\FINDS_task\data\DATASET\nnUNet\nnUNet_results\Dataset503_FINDS\nnUNetTrainer__nnUNetPlans__3d_fullres'
# for folder in os.listdir(json_folder):
#     file_path = os.path.join(json_folder, folder, 'summary.json')  # JSON文件路径
#     output_excel_path = os.path.join(json_folder, folder, 'case_metrics.xlsx')  # 输出Excel文件路径
#     json_data = load_json(file_path)
#     convert_and_save_excel(json_data, output_excel_path)
#     print(f"Saved {output_excel_path}")
