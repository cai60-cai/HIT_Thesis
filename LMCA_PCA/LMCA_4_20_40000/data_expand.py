import os
import pandas as pd
import numpy as np

# 数据路径
input_file = "../LMCA/data/total.csv"
output_file = "data/expanded_total_40000.csv"

# 读取数据
raw_data = pd.read_csv(input_file)

# 统计标签分布
label_counts = raw_data[' Label'].value_counts()

# 定义目标条数
target_count = 40000

def augment_data(data, required_count):
    """对少数类样本进行数据增强，直至达到目标数量"""
    current_count = len(data)
    if current_count >= required_count:
        return data

    # 复制原始数据
    augmented_data = data.copy()

    # 选择数值列
    numerical_columns = data.select_dtypes(include=np.number).columns

    # 插值增加样本
    while len(augmented_data) < required_count:
        # 随机选择两行数据进行插值
        sample1 = data.sample(1)
        sample2 = data.sample(1)
        interpolated = sample1[numerical_columns].reset_index(drop=True) + \
                       sample2[numerical_columns].reset_index(drop=True)
        interpolated = interpolated / 2  # 插值计算
        interpolated[numerical_columns] = interpolated  # 插值只覆盖数值列

        # 保留非数值列的原值
        for col in data.columns:
            if col not in numerical_columns:
                interpolated[col] = sample1[col].values[0]

        # 添加到增强数据中
        augmented_data = pd.concat([augmented_data, interpolated], ignore_index=True)

    # 加噪声增加样本
    while len(augmented_data) < required_count:
        sample = data.sample(1)
        noisy_sample = sample.copy()
        noisy_sample[numerical_columns] += np.random.normal(0, 0.01, size=noisy_sample[numerical_columns].shape)
        augmented_data = pd.concat([augmented_data, noisy_sample], ignore_index=True)

    # 截断到目标数量
    return augmented_data.iloc[:required_count]


# 初始化一个新的DataFrame来存储增强后的数据
expanded_data = pd.DataFrame()

# 遍历每个标签进行数据增强
for label, count in label_counts.items():
    label_data = raw_data[raw_data[' Label'] == label]
    if count < target_count:
        print(f"Augmenting label '{label}' from {count} to {target_count}...")
        label_data = augment_data(label_data, target_count)
    else:
        print(f"Label '{label}' has sufficient samples: {count}.")
    expanded_data = pd.concat([expanded_data, label_data], ignore_index=True)

# 保存增强后的数据
expanded_data.to_csv(output_file, index=False)
print(f"Data augmentation completed. Expanded dataset saved to {output_file}.")
