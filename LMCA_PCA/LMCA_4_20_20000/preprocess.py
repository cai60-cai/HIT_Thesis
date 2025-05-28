import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
from tqdm import tqdm

def load_data(file_path):
    """
    加载CSV格式的数据集
    Args:
        file_path: CSV文件路径
    Returns:
        DataFrame: 加载的数据集
    """
    return pd.read_csv(file_path)

def preprocess_features(df):
    """
    特征预处理，包括标准化等
    Args:
        df: 原始DataFrame
    Returns:
        features: 处理后的特征矩阵
        labels: 处理后的标签
    """
    # 分离特征和标签
    X = df.drop(' Label', axis=1)
    y = df[' Label']
    
    # 标准化特征
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def transform_to_square_batch(features, N=9, batch_size=1000):
    """
    批处理方式将特征转换为方形矩阵(N×N)
    Args:
        features: 原始特征矩阵
        N: 目标矩阵的大小（默认9x9=81 > 78个特征）
        batch_size: 每批处理的样本数
    Returns:
        generator: 生成N×N的特征矩阵批次
    """
    num_samples = features.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        
        batch_features = features[start_idx:end_idx]
        
        # 将78维特征补齐到N*N
        target_size = N * N
        padded_features = np.zeros((end_idx - start_idx, target_size))
        padded_features[:, :features.shape[1]] = batch_features
        
        # 重塑为方形矩阵
        yield padded_features.reshape(-1, N, N)

def encode_labels(labels):
    """
    将标签编码为数字
    Args:
        labels: 原始标签
    Returns:
        numpy array: 编码后的标签
    """
    # 创建标签映射
    unique_labels = labels.unique()
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    
    return labels.map(label_mapping).values, label_mapping

def save_processed_data_batch(generator, labels, indices, label_mapping, output_dir='processed_data', prefix=''):
    """
    分批保存处理后的数据
    Args:
        generator: 特征生成器
        labels: 标签数组
        indices: 样本索引
        label_mapping: 标签映射字典
        output_dir: 输出目录
        prefix: 文件前缀（'train'或'test'）
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存标签映射（仅在保存训练集时）
    if prefix == 'train':
        pd.DataFrame.from_dict(label_mapping, orient='index').to_csv(
            f'{output_dir}/label_mapping.csv'
        )
    
    # 分批保存数据
    for i, batch in enumerate(tqdm(generator, desc=f"保存{prefix}数据")):
        np.save(f'{output_dir}/{prefix}_features_batch_{i}.npy', batch)
    
    # 保存标签
    np.save(f'{output_dir}/{prefix}_labels.npy', labels[indices])
    
    # 保存批次信息
    with open(f'{output_dir}/{prefix}_batch_info.txt', 'w') as f:
        f.write(f'Total batches: {i+1}\n')

def main():
    """
    主函数，执行完整的数据预处理流程
    """
    # 配置参数
    INPUT_FILE = 'data/train_1.csv'  # 输入文件路径
    OUTPUT_DIR = 'processed_data'     # 输出目录
    MATRIX_SIZE = 9                   # N×N矩阵大小 (9x9=81 > 78特征)
    RANDOM_SEED = 42                  # 随机种子
    TEST_SIZE = 0.2                   # 测试集比例
    BATCH_SIZE = 1000                 # 批处理大小
    
    print("加载数据...")
    df = load_data(INPUT_FILE)
    
    print("预处理特征...")
    features, labels = preprocess_features(df)
    
    print("编码标签...")
    encoded_labels, label_mapping = encode_labels(labels)
    
    print("划分训练集和测试集索引...")
    train_indices, test_indices = train_test_split(
        np.arange(len(encoded_labels)), 
        test_size=TEST_SIZE, 
        random_state=RANDOM_SEED,
        stratify=encoded_labels
    )
    
    print(f"处理并保存训练集...")
    train_generator = transform_to_square_batch(
        features[train_indices], 
        N=MATRIX_SIZE, 
        batch_size=BATCH_SIZE
    )
    save_processed_data_batch(
        train_generator, 
        encoded_labels, 
        train_indices,
        label_mapping, 
        OUTPUT_DIR, 
        'train'
    )
    
    print(f"处理并保存测试集...")
    test_generator = transform_to_square_batch(
        features[test_indices], 
        N=MATRIX_SIZE, 
        batch_size=BATCH_SIZE
    )
    save_processed_data_batch(
        test_generator, 
        encoded_labels, 
        test_indices,
        label_mapping, 
        OUTPUT_DIR, 
        'test'
    )
    
    print(f"数据预处理完成！处理后的数据保存在 {OUTPUT_DIR} 目录下")
    print(f"训练集样本数: {len(train_indices)}")
    print(f"测试集样本数: {len(test_indices)}")
    print(f"类别数量: {len(label_mapping)}")

if __name__ == '__main__':
    main()