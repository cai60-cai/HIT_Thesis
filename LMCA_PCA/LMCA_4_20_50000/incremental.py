import pandas as pd

# 数据路径
input_file = "data/balanced_50000.csv"
output_file_1 = "data/incremental_data_1_avg.csv"
output_file_2 = "data/train_1_avg.csv"

# 读取数据
data = pd.read_csv(input_file)

# 指定需要分离出来的两类标签
incremental_labels = ["Web Attack � Brute Force", "Web Attack � Sql Injection","Web Attack � XSS","Bot"]  # 替换为您希望的两类

# 检查是否标签存在于数据集中
all_labels = data[' Label'].unique()
for label in incremental_labels:
    if label not in all_labels:
        raise ValueError(f"Label '{label}' not found in the dataset.")

# 按标签分离数据
incremental_data = data[data[' Label'].isin(incremental_labels)]
remaining_data = data[~data[' Label'].isin(incremental_labels)]

# 保存两部分数据
incremental_data.to_csv(output_file_1, index=False)
remaining_data.to_csv(output_file_2, index=False)

print(f"Incremental data saved to {output_file_1} with labels: {incremental_labels}.")
print(f"Remaining data saved to {output_file_2}.")
