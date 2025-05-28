import os
import pandas as pd
import numpy as np

# 数据路径
input_file = "../LMCA/data/total.csv"
output_file = "data/expanded_total_40000_avg.csv"

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
        interpolated = sample1[numerical_columns].values + np.random.rand() * (sample2[numerical_columns].values - sample1[numerical_columns].values)
        interpolated_data = sample1.copy()
        interpolated_data[numerical_columns] = interpolated

        # 将插值后的数据加入到增强数据集中
        augmented_data = pd.concat([augmented_data, interpolated_data])

    # 返回增强后的数据
    return augmented_data

def balance_data(data, target_count, normal_class='BENIGN'):
    """减少 normal 类样本数量，使数据集更加平衡"""
    # 选择 normal 类样本
    normal_data = data[data[' Label'] == normal_class]
    
    # 选择非 normal 类样本
    non_normal_data = data[data[' Label'] != normal_class]
    
    # 计算目标数量
    normal_target_count = target_count // len(label_counts)
    
    # 随机抽样减少 normal 类样本
    normal_data = normal_data.sample(normal_target_count)

    # 对少数类样本进行增强
    augmented_data = []
    for label in non_normal_data[' Label'].unique():
        class_data = non_normal_data[non_normal_data[' Label'] == label]
        augmented_class_data = augment_data(class_data, target_count // len(label_counts))
        augmented_data.append(augmented_class_data)

    # 合并正常类和增强后的非正常类数据
    balanced_data = pd.concat([normal_data] + augmented_data)

    # 打乱数据顺序
    balanced_data = balanced_data.sample(frac=1).reset_index(drop=True)

    return balanced_data

# 平衡数据
balanced_data = balance_data(raw_data, target_count)

# 保存到文件
balanced_data.to_csv(output_file, index=False)

print(f"数据增强并平衡后，保存到 {output_file}")
