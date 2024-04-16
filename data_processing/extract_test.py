import pandas as pd

# Load the CSV file
file_path = r"Z:\Nana\TransUnet_For_FINDS\data\data_label_pairs_compare_240124.csv"  # Replace 'path_to_your_file.csv' with the actual file path
data = pd.read_csv(file_path)

# Identify rows where "Label Path 1" is None and "Label Path 2" is not None
# Assuming 'None' might be represented as a string or actual NaN in the dataframe
filtered_rows = data[(data['Label Path 1'].isnull() | (data['Label Path 1'] == 'None')) & (~data['Label Path 2'].isnull() & (data['Label Path 2'] != 'None'))]

# Extract the "num" values for these rows
num_values = filtered_rows['num'].tolist()

# Count the number of such "num" values
num_count = len(num_values)

# Display the results
print("Num values where 'Label Path 1' is None and 'Label Path 2' is not None:", num_values)
print("Count of such num values:", num_count)

