import h5py
import numpy as np
from scipy import stats
import pandas as pd
from scipy.stats import ttest_ind, t
from statsmodels.stats.weightstats import ttost_ind
import math
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW
import statsmodels.stats.api as sms

def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / math.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

# 假设等价性边界，需要根据具体情况调整
low_eq_bound = -0.2
high_eq_bound = 0.2

# File path to the HDF5 data
file_path = 'Task503_features_20240329.h5'

# Label names mapping
label_names = {
    'Label_1': 'Left salivary glands',
    'Label_2': 'Right salivary glands',
    'Label_3': 'Left wing disc',
    'Label_4': 'Right wing disc',
    'Label_5': 'Left haltere disc',
    'Label_6': 'Right haltere disc',
}

# Label pairs for comparison
label_pairs = [
    ('Label_1', 'Label_2'),
    ('Label_3', 'Label_4'),
    ('Label_5', 'Label_6'),
]

# Selected attributes for t-test
selected_attributes = [
    'area', 'area_convex', 'area_filled', 'axis_major_length', 'axis_minor_length',
    'equivalent_diameter_area', 'euler_number', 'extent', 'feret_diameter_max',
    'intensity_max', 'intensity_mean', 'intensity_min', 'solidity'
]

# Prepare to collect results with error handling
results_with_error_handling = []

# Open the HDF5 file and process each attribute for each label pair
with h5py.File(file_path, 'r') as file:
    for label1, label2 in label_pairs:
        for attr in selected_attributes:
            try:
                if attr in file[label1] and attr in file[label2]:
                    data1 = file[label1][attr][:]
                    data2 = file[label2][attr][:]

                    if data1.ndim == 1 and data2.ndim == 1 and not (
                            np.isnan(data1).any() or np.isinf(data1).any() or np.isnan(data2).any() or np.isinf(
                            data2).any()):
                        t_stat, p_value = stats.ttest_ind(data1, data2)
                        effect_size = cohen_d(data1, data2)

                        # 计算置信区间
                        cm = sms.CompareMeans(sms.DescrStatsW(data1), sms.DescrStatsW(data2))
                        ci_low, ci_high = cm.tconfint_diff(usevar='unequal')

                        # 进行等价性测试
                        tost_p1, tost_p2, tost = ttost_ind(data1, data2, low_eq_bound, high_eq_bound)

                        mean1 = np.mean(data1)
                        mean2 = np.mean(data2)
                        median1 = np.median(data1)
                        median2 = np.median(data2)


                        results_with_error_handling.append({
                            'Label Pair': f'{label_names[label1]} vs {label_names[label2]}',
                            'Attribute': attr,
                            'Mean (Left)': mean1,
                            'Mean (Right)': mean2,
                            'Median (Left)': median1,
                            'Median (Right)': median2,
                            'Effect Size (Cohen\'s d)': effect_size,
                            'T-Statistic': t_stat,
                            'P-Value': p_value,
                            'CI Low': ci_low,
                            'CI High': ci_high,
                            'TOST P1': tost_p1,
                            'TOST P2': tost_p2,
                            'TOST': tost
                        })

                    else:
                        raise ValueError("Data is not 1D or contains NaN/Inf values")
            except Exception as e:
                print(f"Error processing {attr} for {label1} vs {label2}: {e}")

# Convert results to DataFrame
results_df = pd.DataFrame(results_with_error_handling)

# Path to save the Excel file
excel_path = 't_test_results_20240410.xlsx'

# Save results to Excel
results_df.to_excel(excel_path, index=False)

print(f"Results saved to {excel_path}")
