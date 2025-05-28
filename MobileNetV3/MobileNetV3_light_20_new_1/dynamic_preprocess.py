
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

def create_processed_folders(base_dir='pca', dimensions=[10, 20, 30, 40, 50, 60, 70, 78]):
    """为每个维度创建processed_data文件夹"""
    for dim in dimensions:
        processed_dir = os.path.join(base_dir, f'dim_{dim}', 'incremental_processed_data')
        os.makedirs(processed_dir, exist_ok=True)

def load_pca_data(dim, base_dir='pca'):
    """加载特定维度的PCA数据"""
    dim_dir = os.path.join(base_dir, f'dim_{dim}')
    
    # 加载特征和标签
    features = np.load(os.path.join(dim_dir, 'features.npy'))
    labels = np.load(os.path.join(dim_dir, 'labels.npy'), allow_pickle=True)
    
    return features, labels

def transform_to_square_batch(features, N, batch_size=1000):
    """将特征转换为方形矩阵，批处理方式"""
    num_samples = features.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        
        batch_features = features[start_idx:end_idx]
        
        # 将特征补齐到N*N
        target_size = N * N
        padded_features = np.zeros((end_idx - start_idx, target_size))
        padded_features[:, :features.shape[1]] = batch_features
        
        # 重塑为方形矩阵
        yield padded_features.reshape(-1, N, N)

def encode_labels(labels):
    """将标签编码为数字"""
    # 确保标签是字符串类型
    labels = labels.astype(str)
    unique_labels = np.unique(labels)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    
    encoded = np.array([label_mapping[label] for label in labels], dtype=np.int64)
    return encoded, label_mapping

def save_processed_data_batch(generator, labels, indices, label_mapping, output_dir, prefix=''):
    """分批保存处理后的数据"""
    # 保存标签映射（仅在保存训练集时）
    if prefix == 'train':
        pd.DataFrame.from_dict(label_mapping, orient='index').to_csv(
            f'{output_dir}/label_mapping.csv'
        )
    
    # 分批保存数据
    batch_count = 0
    for i, batch in enumerate(tqdm(generator, desc=f"保存{prefix}数据")):
        np.save(f'{output_dir}/{prefix}_features_batch_{i}.npy', batch)
        batch_count = i + 1
    
    # 保存标签 (确保标签是整数类型)
    np.save(f'{output_dir}/{prefix}_labels.npy', labels[indices].astype(np.int64))
    
    # 保存批次信息
    with open(f'{output_dir}/{prefix}_batch_info.txt', 'w') as f:
        f.write(f'Total batches: {batch_count}\n')

def process_dimension(dim, test_size=0.2, random_seed=42, base_dir='pca'):
    """处理特定维度的数据"""
    print(f"\nProcessing dimension: {dim}")
    
    # 计算需要的方形矩阵大小
    N = int(np.ceil(np.sqrt(dim)))
    print(f"Using {N}x{N} matrix size for dimension {dim}")
    
    # 加载该维度的PCA数据
    features, labels = load_pca_data(dim, base_dir)
    
    # 编码标签
    encoded_labels, label_mapping = encode_labels(labels)
    
    # 划分训练集和测试集
    train_indices, test_indices = train_test_split(
        np.arange(len(encoded_labels)), 
        test_size=test_size, 
        random_state=random_seed,
        stratify=encoded_labels
    )
    
    # 创建保存目录
    output_dir = os.path.join(base_dir, f'dim_{dim}', 'processed_data_incremental')
    # output_dir = 'processed_data_drift'
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理训练集
    print("Processing training data...")
    train_generator = transform_to_square_batch(
        features[train_indices], N=N
    )
    save_processed_data_batch(
        train_generator,
        encoded_labels,
        train_indices,
        label_mapping,
        output_dir,
        'train'
    )
    
    # 处理测试集
    print("Processing test data...")
    test_generator = transform_to_square_batch(
        features[test_indices], N=N
    )
    save_processed_data_batch(
        test_generator,
        encoded_labels,
        test_indices,
        label_mapping,
        output_dir,
        'test'
    )
    
    # 保存维度信息
    with open(os.path.join(output_dir, 'dimension_info.txt'), 'w') as f:
        f.write(f'Original dimension: {dim}\n')
        f.write(f'Matrix size: {N}x{N}\n')
        f.write(f'Training samples: {len(train_indices)}\n')
        f.write(f'Test samples: {len(test_indices)}\n')
        f.write(f'Number of classes: {len(label_mapping)}\n')

def main():
    # 配置参数
    base_dir = 'pca/pca_incremental_data_last_5_3'
    dimensions = [20]
    test_size = 0.2
    random_seed = 42
    
    # 创建必要的文件夹
    create_processed_folders(base_dir, dimensions)
    
    # 处理每个维度
    for dim in dimensions:
        process_dimension(dim, test_size, random_seed, base_dir)
        print(f"Completed processing dimension {dim}\n")
    
    print("All dimensions processed successfully!")

if __name__ == '__main__':
    main()
