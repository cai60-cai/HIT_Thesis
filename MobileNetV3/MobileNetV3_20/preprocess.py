
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
from tqdm import tqdm

def load_data(file_path):
    """加载CSV数据"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Total samples: {len(df)}")
    return df

def preprocess_features(df):
    """特征预处理"""
    # 分离特征和标签
    X = df.drop(' Label', axis=1)
    y = df[' Label']
    
    # 特征标准化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def features_to_matrix(features):
    """将特征转换为方形矩阵"""
    # 计算需要的矩阵大小
    n_features = features.shape[1]
    matrix_size = int(np.ceil(np.sqrt(n_features)))
    
    # 创建方形矩阵
    matrix_features = []
    for sample in features:
        # 填充到平方大小
        padded = np.zeros(matrix_size * matrix_size)
        padded[:n_features] = sample
        # 重塑为方形矩阵
        matrix = padded.reshape(matrix_size, matrix_size)
        matrix_features.append(matrix)
    
    return np.array(matrix_features)

def encode_labels(labels):
    """标签编码"""
    unique_labels = labels.unique()
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    encoded = np.array([label_mapping[label] for label in labels])
    return encoded, label_mapping

def save_batch(data, labels, indices, output_dir, prefix):
    """分批保存数据"""
    batch_size = 1000
    num_samples = len(indices)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        batch_data = data[batch_indices]
        np.save(f"{output_dir}/{prefix}_features_batch_{i}.npy", batch_data)
    
    # 保存标签
    np.save(f"{output_dir}/{prefix}_labels.npy", labels[indices])

def main():
    # 配置参数
    input_file = 'data/train_1.csv'
    output_dir = 'processed_data'
    test_size = 0.2
    random_seed = 42
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载和预处理数据
    df = load_data(input_file)
    X, y = preprocess_features(df)
    
    # 转换为方形矩阵
    print("Converting features to square matrices...")
    X_matrix = features_to_matrix(X)
    print(f"Matrix shape: {X_matrix.shape}")
    
    # 编码标签
    print("Encoding labels...")
    y_encoded, label_mapping = encode_labels(y)
    
    # 保存标签映射
    pd.DataFrame.from_dict(label_mapping, orient='index').to_csv(
        f'{output_dir}/label_mapping.csv'
    )
    
    # 划分训练集和测试集
    print("Splitting into train and test sets...")
    train_indices, test_indices = train_test_split(
        np.arange(len(y_encoded)),
        test_size=test_size,
        random_state=random_seed,
        stratify=y_encoded
    )
    
    # 保存数据
    print("Saving training data...")
    save_batch(X_matrix, y_encoded, train_indices, output_dir, 'train')
    
    print("Saving test data...")
    save_batch(X_matrix, y_encoded, test_indices, output_dir, 'test')
    
    # 保存数据集信息
    info = {
        'num_features': X.shape[1],
        'matrix_size': X_matrix.shape[1],
        'num_classes': len(label_mapping),
        'train_samples': len(train_indices),
        'test_samples': len(test_indices)
    }
    
    # 打印数据集信息
    print("\nDataset Information:")
    print(f"Number of features: {info['num_features']}")
    print(f"Matrix size: {info['matrix_size']}x{info['matrix_size']}")
    print(f"Number of classes: {info['num_classes']}")
    print(f"Training samples: {info['train_samples']}")
    print(f"Test samples: {info['test_samples']}")
    
    np.save(f'{output_dir}/dataset_info.npy', info)

if __name__ == '__main__':
    main()
