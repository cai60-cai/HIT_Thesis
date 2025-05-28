import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class FeatureDriftGenerator:
    """特征漂移生成器"""
    def __init__(self, noise_level=0.1, scale_range=(0.9, 1.1)):
        self.noise_level = noise_level
        self.scale_range = scale_range
        
    def add_gaussian_noise(self, features):
        """添加高斯噪声"""
        noise = np.random.normal(0, self.noise_level * np.std(features, axis=0), features.shape)
        return features + noise
    
    def scale_features(self, features):
        """特征缩放"""
        scales = np.random.uniform(self.scale_range[0], self.scale_range[1], features.shape[1])
        return features * scales

def generate_drift_data():
    """生成漂移数据并评估"""
    # 配置
    base_dir = 'pca/pca_org/dim_20'
    output_dir = 'drift_evaluation_results_D'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载原始数据
    print("Loading data...")
    features = np.load(os.path.join(base_dir, 'features.npy'))
    labels = np.load(os.path.join(base_dir, 'labels.npy'), allow_pickle=True)
    
    print(f"Data loaded. Shape: {features.shape}, Labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")
    
    # 定义不同的类别组
    dos_ddos_classes = ['DoS slowloris','DDoS','DoS Hulk']
    noise_classes = ['BENIGN', 'PortScan', 'SSH-Patator', 'Infiltration']
    
    # 创建漂移生成器
    drift_generator = FeatureDriftGenerator(
        noise_level=0.05,   
        scale_range=(0.95, 1.05)  
    )
    
    # 生成漂移数据
    print("Generating drift data...")
    drifted_features = features.copy()
    
    drift_ratio = 1.0
    total_drifted = 0
    
    # 预先创建drift_config
    drift_config = {
        'dos_ddos_classes': list(dos_ddos_classes),
        'noise_classes': list(noise_classes),
        'noise_level': 0.4,
        'scale_range': [0.6, 1.4],
        'num_samples': int(len(features)),
        'drift_ratio': drift_ratio,
        'drifted_samples_per_class': {}
    }
    
    # 对 DoS 和 DDoS 类别应用特征缩放
    for class_name in dos_ddos_classes:
        class_mask = labels == class_name
        class_features = drifted_features[class_mask]
        
        # 随机选择部分样本进行缩放
        num_samples = class_features.shape[0]
        drift_samples_num = int(num_samples * drift_ratio)
        drift_indices = np.random.choice(num_samples, drift_samples_num, replace=False)
        
        class_features[drift_indices] = drift_generator.scale_features(class_features[drift_indices])
        drifted_features[class_mask] = class_features
        
        total_drifted += drift_samples_num
        drift_config['drifted_samples_per_class'][str(class_name)] = int(drift_samples_num)
    
    # 对 BENIGN、PortScan 等类别添加高斯噪声
    for class_name in noise_classes:
        class_mask = labels == class_name
        class_features = drifted_features[class_mask]
        
        # 随机选择部分样本添加噪声
        num_samples = class_features.shape[0]
        drift_samples_num = int(num_samples * drift_ratio)
        drift_indices = np.random.choice(num_samples, drift_samples_num, replace=False)
        
        class_features[drift_indices] = drift_generator.add_gaussian_noise(class_features[drift_indices])
        drifted_features[class_mask] = class_features
        
        total_drifted += drift_samples_num
        drift_config['drifted_samples_per_class'][str(class_name)] = int(drift_samples_num)
    
    drift_config['num_drifted_samples'] = total_drifted
    print(f"Total drifted samples: {total_drifted}")
    
    # 保存漂移数据
    drift_dir = 'pca/pca_drift_D_2/dim_20'
    os.makedirs(drift_dir, exist_ok=True)
    np.save(os.path.join(drift_dir, 'features.npy'), drifted_features)
    np.save(os.path.join(drift_dir, 'labels.npy'), labels)
    
    # 保存配置信息
    with open(os.path.join(output_dir, 'drift_config.json'), 'w') as f:
        json.dump(drift_config, f, indent=4)

    # ... 后续代码保持不变 ...

    # 绘制特征分布对比图
    for class_name in dos_ddos_classes + noise_classes:
        class_mask = labels == class_name
        if np.sum(class_mask) == 0:
            continue
            
        plt.figure(figsize=(15, 5))
        
        # 只展示前3个主要特征
        for feature_idx in range(3):
            plt.subplot(1, 3, feature_idx + 1)
            
            # 计算适当的bins范围
            feature_range = (
                min(features[class_mask, feature_idx].min(), drifted_features[class_mask, feature_idx].min()),
                max(features[class_mask, feature_idx].max(), drifted_features[class_mask, feature_idx].max())
            )
            
            plt.hist(features[class_mask, feature_idx], bins=50, range=feature_range, alpha=0.5, label='Original', density=True)
            plt.hist(drifted_features[class_mask, feature_idx], bins=50, range=feature_range, alpha=0.5, label='Drifted', density=True)
            
            plt.title(f'{class_name}\nFeature {feature_idx}')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'class_{class_name}_distribution.png'))
        plt.close()

if __name__ == '__main__':
    generate_drift_data()
