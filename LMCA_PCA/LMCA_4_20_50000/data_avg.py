import os
import pandas as pd
import numpy as np
def augment_data(data, required_count):
    """对少数类样本进行多种方式的数据增强"""
    current_count = len(data)
    if current_count >= required_count:
        return data
    
    augmented_data = data.copy()
    numerical_columns = data.select_dtypes(include=np.number).columns
    
    while len(augmented_data) < required_count:
        # 1. SMOTE类似的插值
        if len(augmented_data) < required_count:
            samples = data.sample(n=2)
            alpha = np.random.random()
            interpolated = samples.iloc[0][numerical_columns] * alpha + \
                         samples.iloc[1][numerical_columns] * (1-alpha)
            new_sample = samples.iloc[0].copy()
            new_sample[numerical_columns] = interpolated
            augmented_data = pd.concat([augmented_data, pd.DataFrame([new_sample])], 
                                     ignore_index=True)
        
        # 2. 添加高斯噪声
        if len(augmented_data) < required_count:
            sample = data.sample(1)
            noise = np.random.normal(0, 0.01, size=(1, len(numerical_columns)))
            noisy_sample = sample.copy()
            noisy_sample[numerical_columns] += noise
            augmented_data = pd.concat([augmented_data, noisy_sample], 
                                     ignore_index=True)
        
        # 3. 特征缩放扰动
        if len(augmented_data) < required_count:
            sample = data.sample(1)
            scale = np.random.uniform(0.95, 1.05, size=(1, len(numerical_columns)))
            scaled_sample = sample.copy()
            scaled_sample[numerical_columns] *= scale
            augmented_data = pd.concat([augmented_data, scaled_sample], 
                                     ignore_index=True)
    
    # 确保数值合理
    for col in numerical_columns:
        min_val = data[col].min()
        max_val = data[col].max()
        augmented_data[col] = augmented_data[col].clip(min_val, max_val)
    
    return augmented_data.iloc[:required_count]


def process_in_batches(input_file, output_file, target_count, batch_size=10000):
    """分批处理大数据集"""
    # 读取原始数据统计信息
    labels_stats = pd.read_csv(input_file)[' Label'].value_counts()
    
    # 分批处理每个标签
    for label in labels_stats.index:
        print(f"\nProcessing label: {label}")
        
        # 读取该标签的所有数据
        chunk_iterator = pd.read_csv(input_file, chunksize=batch_size)
        label_data = pd.DataFrame()
        
        for chunk in chunk_iterator:
            label_chunk = chunk[chunk[' Label'] == label]
            label_data = pd.concat([label_data, label_chunk])
        
        # 数据增强
        if len(label_data) < target_count:
            print(f"Augmenting from {len(label_data)} to {target_count}")
            augmented = augment_data(label_data, target_count)
            
            # 追加写入文件
            augmented.to_csv(output_file, mode='a', header=not os.path.exists(output_file),
                           index=False)
        else:
            # 对于样本足够的类别，随机采样到目标数量
            sampled = label_data.sample(n=target_count, random_state=42)
            sampled.to_csv(output_file, mode='a', header=not os.path.exists(output_file),
                          index=False)
            
if __name__ == '__main__':
    input_file = "../LMCA/data/total.csv"
    output_file = "data/balanced_50000.csv"
    target_count = 50000  # 每个类别的目标数量
    
    print("Starting data balancing process...")
    process_in_batches(input_file, output_file, target_count)
    
    # 验证结果
    final_data = pd.read_csv(output_file)
    print("\nFinal class distribution:")
    print(final_data[' Label'].value_counts())