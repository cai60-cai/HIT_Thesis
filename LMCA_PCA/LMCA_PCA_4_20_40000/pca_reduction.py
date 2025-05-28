
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import os
from tqdm import tqdm

def create_dimension_folders(base_dir='pca'):
    """创建不同维度的文件夹"""
    dimensions = [10, 20, 30, 40, 50, 60, 70, 78]
    folders = {}
    
    for dim in dimensions:
        folder_path = os.path.join(base_dir, f'dim_{dim}')
        os.makedirs(folder_path, exist_ok=True)
        folders[dim] = folder_path
    
    return folders, dimensions

def load_and_preprocess_data(file_path):
    """加载和预处理数据"""
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # 分离特征和标签
    X = df.drop(' Label', axis=1)
    y = df[' Label']
    
    # 标准化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def perform_pca_reduction(X, dimensions):
    """对每个目标维度执行PCA降维"""
    pca_results = {}
    
    for dim in tqdm(dimensions, desc="Performing PCA"):
        pca = PCA(n_components=dim)
        X_reduced = pca.fit_transform(X)
        explained_variance = pca.explained_variance_ratio_.sum()
        
        pca_results[dim] = {
            'data': X_reduced,
            'explained_variance': explained_variance,
            'pca_model': pca
        }
        
    return pca_results

def save_dimension_data(pca_results, y, folders):
    """保存每个维度的数据"""
    for dim, result in pca_results.items():
        folder_path = folders[dim]
        
        # 保存降维后的特征
        np.save(os.path.join(folder_path, 'features.npy'), result['data'])
        
        # 保存标签
        np.save(os.path.join(folder_path, 'labels.npy'), y.values)
        
        # 保存PCA模型
        np.save(os.path.join(folder_path, 'pca_model.npy'), 
               {'components': result['pca_model'].components_,
                'mean': result['pca_model'].mean_,
                'explained_variance': result['explained_variance']})
        
        # 保存降维信息
        with open(os.path.join(folder_path, 'pca_info.txt'), 'w') as f:
            f.write(f"Dimension: {dim}\n")
            f.write(f"Explained variance ratio: {result['explained_variance']:.4f}\n")

def main():
    # 创建基础目录
    base_dir = 'pca'
    os.makedirs(base_dir, exist_ok=True)
    
    # 创建维度文件夹
    folders, dimensions = create_dimension_folders(base_dir)
    
    # 加载和预处理数据
    X, y = load_and_preprocess_data('data/train_1.csv')
    
    # 执行PCA降维
    pca_results = perform_pca_reduction(X, dimensions)
    
    # 保存结果
    save_dimension_data(pca_results, y, folders)
    
    print("\nPCA reduction completed!")
    print("\nExplained variance ratios:")
    for dim in dimensions:
        print(f"Dimension {dim}: {pca_results[dim]['explained_variance']:.4f}")

if __name__ == '__main__':
    main()



# Explained variance ratios:
# Dimension 10: 0.9479
# Dimension 20: 0.9966
# Dimension 30: 0.9993
# Dimension 40: 0.9999
# Dimension 50: 1.0000
# Dimension 60: 1.0000
# Dimension 70: 1.0000
# Dimension 78: 1.0000