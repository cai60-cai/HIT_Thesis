import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import Counter
import time

class FastTrafficAugmentor:
    def __init__(self, data, label_column='Label'):
        self.data = data
        self.label_column = label_column
        self.numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        self.numeric_columns = self.numeric_columns.drop(label_column) if label_column in self.numeric_columns else self.numeric_columns
        
        # 预先提取时间和包相关的列
        self.time_columns = [col for col in self.numeric_columns if 
                           any(x in col.lower() for x in ['duration', 'iat', 'time'])]
        self.packet_columns = [col for col in self.numeric_columns if 
                             any(x in col.lower() for x in ['packet', 'length', 'bytes'])]

    def batch_augment(self, samples, batch_size=1000):
        """批量增强数据"""
        # 1. 高斯噪声 (向量化操作)
        noise = np.random.normal(0, 0.05, (batch_size, len(self.numeric_columns)))
        augmented = samples[self.numeric_columns].values * (1 + noise)
        
        # 2. 时间扰动
        if self.time_columns:
            time_warp = np.random.uniform(0.9, 1.1, (batch_size, 1))
            augmented[:, [samples.columns.get_loc(col) for col in self.time_columns]] *= time_warp
            
        # 3. 包特征修改
        if self.packet_columns:
            packet_modify = np.random.uniform(0.9, 1.1, (batch_size, 1))
            augmented[:, [samples.columns.get_loc(col) for col in self.packet_columns]] *= packet_modify
        
        # 确保所有值非负
        augmented = np.maximum(augmented, 0)
        
        # 创建增强后的DataFrame
        augmented_df = pd.DataFrame(augmented, columns=self.numeric_columns)
        augmented_df[self.label_column] = samples[self.label_column].values
        
        return augmented_df

    def augment_data(self, augment_ratio=1.0):
        """快速数据增强"""
        start_time = time.time()
        
        # 计算需要生成的样本数
        samples_to_generate = int(len(self.data) * augment_ratio)
        
        # 统计原始数据分布
        original_dist = Counter(self.data[self.label_column])
        print("\nOriginal data distribution:")
        for label, count in original_dist.items():
            print(f"Class '{label}': {count} samples")
        
        # 批量处理
        batch_size = min(1000, samples_to_generate)
        num_batches = (samples_to_generate + batch_size - 1) // batch_size
        
        augmented_samples = []
        for i in range(num_batches):
            current_batch_size = min(batch_size, samples_to_generate - i * batch_size)
            # 随机选择原始样本
            indices = np.random.randint(0, len(self.data), current_batch_size)
            batch_samples = self.data.iloc[indices]
            # 增强当前批次
            augmented_batch = self.batch_augment(batch_samples, current_batch_size)
            augmented_samples.append(augmented_batch)
            
            # 打印进度
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{num_batches} batches...")
        
        # 合并所有增强数据
        augmented_data = pd.concat([self.data] + augmented_samples, ignore_index=True)
        
        # 统计增强后的分布
        augmented_dist = Counter(augmented_data[self.label_column])
        print("\nAugmented data distribution:")
        for label, count in augmented_dist.items():
            print(f"Class '{label}': {count} samples")
        
        print(f"\nTime taken: {time.time() - start_time:.2f} seconds")
        return augmented_data

def main():
    # 读取数据
    start_time = time.time()
    print("Loading data...")
    data = pd.read_csv('data/incremental_data_3.csv')  # 替换为实际路径
    
    print(f"\nOriginal data shape: {data.shape}")
    print(f"Label column: {data.columns[-1]}")
    
    # 创建增强器并增强数据
    augmentor = FastTrafficAugmentor(data, label_column=data.columns[-1])
    augmented_data = augmentor.augment_data(augment_ratio=4.0)
    
    # 保存结果
    output_file = 'data/augmented_traffic_data_last_5_3.csv'
    augmented_data.to_csv(output_file, index=False)
    
    print(f"\nFinal data shape: {augmented_data.shape}")
    print(f"Data saved to: {output_file}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    main()