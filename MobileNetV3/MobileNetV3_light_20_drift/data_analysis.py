
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from train import TrafficDataset
import torch
import os
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataAnalyzer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.dataset = TrafficDataset(data_dir, 'train')
        self.features = self.dataset.features
        self.labels = self.dataset.labels
        
# 在此处对特征进行展平
        if len(self.features.shape) > 2:
            # 将多维特征映射为二维
            self.features = self.features.reshape(self.features.shape[0], -1)

    def analyze_class_distribution(self):
        """分析类别分布"""
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        
        # 创建饼图
        plt.figure(figsize=(10, 8))
        plt.pie(counts, labels=[f'Class {l}' for l in unique_labels], autopct='%1.1f%%')
        plt.title('Class Distribution')
        plt.savefig(os.path.join(self.data_dir, 'class_distribution.png'))
        plt.close()
        
        # 打印详细统计
        print("\nClass Distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"Class {label}: {count} samples ({count/len(self.labels)*100:.2f}%)")
    
    def analyze_feature_statistics(self):
        """分析每个类别的特征统计信息"""
        unique_labels = np.unique(self.labels)
        
        stats = {}
        for class_id in unique_labels:
            class_features = self.features[self.labels == class_id]
            stats[f'Class_{class_id}'] = {
                'mean': np.mean(class_features, axis=0),
                'std': np.std(class_features, axis=0),
                'min': np.min(class_features, axis=0),
                'max': np.max(class_features, axis=0)
            }
        
        # 保存统计信息
        with open(os.path.join(self.data_dir, 'feature_statistics.txt'), 'w') as f:
            for class_name, class_stats in stats.items():
                f.write(f"\n{class_name}:\n")
                f.write(f"Mean: {class_stats['mean'][:5]}...\n")
                f.write(f"Std: {class_stats['std'][:5]}...\n")
                f.write(f"Min: {class_stats['min'][:5]}...\n")
                f.write(f"Max: {class_stats['max'][:5]}...\n")
                f.write("-" * 50 + "\n")
        
        return stats
    
    def visualize_feature_distributions(self, num_features=5):
        """可视化特征分布"""
        unique_labels = np.unique(self.labels)
        
        plt.figure(figsize=(15, 3*num_features))
        for i in range(num_features):
            plt.subplot(num_features, 1, i+1)
            for class_id in unique_labels:
                class_features = self.features[self.labels == class_id]
                sns.kdeplot(class_features[:, i], label=f'Class {class_id}')
            plt.title(f'Feature {i+1} Distribution')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'feature_distributions.png'))
        plt.close()
    
    def analyze_pca_components(self):
        """分析PCA成分"""
        # 标准化数据
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.features)
        
        # 执行PCA
        pca = PCA()
        pca_result = pca.fit_transform(features_scaled)
        
        # 计算解释方差比
        explained_var_ratio = pca.explained_variance_ratio_
        cumulative_var_ratio = np.cumsum(explained_var_ratio)
        
        # 绘制解释方差图
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(explained_var_ratio) + 1), explained_var_ratio, 'bo-')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Scree Plot')
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(cumulative_var_ratio) + 1), cumulative_var_ratio, 'ro-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Cumulative Explained Variance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'pca_analysis.png'))
        plt.close()
        
        # 打印主要成分的贡献
        print("\nPCA Analysis:")
        for i, ratio in enumerate(explained_var_ratio[:10], 1):
            print(f"PC{i}: {ratio*100:.2f}% variance explained")
    
    def analyze_feature_correlations(self):
        """分析特征相关性"""
        # 计算相关性矩阵
        correlation_matrix = np.corrcoef(self.features.T)
        
        # 绘制热图
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.savefig(os.path.join(self.data_dir, 'feature_correlations.png'))
        plt.close()
        
        # 找出高度相关的特征对
        high_correlation_threshold = 0.8
        high_correlations = []
        
        for i in range(len(correlation_matrix)):
            for j in range(i+1, len(correlation_matrix)):
                if abs(correlation_matrix[i,j]) > high_correlation_threshold:
                    high_correlations.append((i, j, correlation_matrix[i,j]))
        
        # 保存高相关性特征对
        with open(os.path.join(self.data_dir, 'high_correlations.txt'), 'w') as f:
            f.write("Highly correlated features (|correlation| > 0.8):\n")
            for i, j, corr in sorted(high_correlations, key=lambda x: abs(x[2]), reverse=True):
                f.write(f"Feature {i} and Feature {j}: {corr:.3f}\n")
    
    def run_full_analysis(self):
        """运行完整分析"""
        print("Starting comprehensive data analysis...")
        
        print("\n1. Analyzing class distribution...")
        self.analyze_class_distribution()
        
        print("\n2. Computing feature statistics...")
        self.analyze_feature_statistics()
        
        print("\n3. Visualizing feature distributions...")
        self.visualize_feature_distributions()
        
        print("\n4. Analyzing PCA components...")
        self.analyze_pca_components()
        
        print("\n5. Analyzing feature correlations...")
        self.analyze_feature_correlations()
        
        print("\nAnalysis complete. Results saved in the data directory.")

def main():
    # 设置数据目录
    data_dir = 'processed_data_incremental_last'  # 替换为你的数据目录
    
    # 创建分析器并运行分析
    analyzer = DataAnalyzer(data_dir)
    analyzer.run_full_analysis()

if __name__ == '__main__':
    main()
