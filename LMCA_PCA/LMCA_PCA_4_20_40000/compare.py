import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import time
from sklearn.preprocessing import LabelEncoder

# 加载数据集（假设为CSV格式）
df = pd.read_csv('data/train_1.csv')

# 随机选择5%的样本
df_sampled = df.sample(frac=0.005, random_state=42)

# 特征预处理
features = df_sampled.drop(' Label', axis=1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 将标签编码为数值
label_encoder = LabelEncoder()
df_sampled['Label_encoded'] = label_encoder.fit_transform(df_sampled[' Label'])

# PCA降维
start_time = time.time()
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_scaled)
pca_time = time.time() - start_time

# 计算PCA解释的方差比率
explained_variance = pca.explained_variance_ratio_

# t-SNE降维
start_time_tsne = time.time()
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(features_scaled)
tsne_time = time.time() - start_time_tsne

# UMAP降维
start_time_umap = time.time()
umap_model = umap.UMAP(n_components=2, random_state=42)
umap_result = umap_model.fit_transform(features_scaled)
umap_time = time.time() - start_time_umap

# KMeans聚类评估（用于t-SNE和UMAP的效果度量）
def cluster_and_score(result, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(result)
    score = silhouette_score(result, kmeans.labels_)
    return score

# KMeans聚类的轮廓系数
pca_score = cluster_and_score(pca_result)
tsne_score = cluster_and_score(tsne_result)
umap_score = cluster_and_score(umap_result)

# 打印降维结果和时间
print(f"PCA降维完成，耗时: {pca_time:.4f}秒")
print(f"PCA解释的方差比率: {explained_variance}")
print(f"PCA聚类轮廓系数: {pca_score:.4f}")

print(f"t-SNE降维完成，耗时: {tsne_time:.4f}秒")
print(f"t-SNE聚类轮廓系数: {tsne_score:.4f}")

print(f"UMAP降维完成，耗时: {umap_time:.4f}秒")
print(f"UMAP聚类轮廓系数: {umap_score:.4f}")

# 绘制降维结果
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# PCA
ax[0].scatter(pca_result[:, 0], pca_result[:, 1], c=df_sampled['Label_encoded'], cmap='viridis')
ax[0].set_title('PCA')

# t-SNE
ax[1].scatter(tsne_result[:, 0], tsne_result[:, 1], c=df_sampled['Label_encoded'], cmap='viridis')
ax[1].set_title('t-SNE')

# UMAP
ax[2].scatter(umap_result[:, 0], umap_result[:, 1], c=df_sampled['Label_encoded'], cmap='viridis')
ax[2].set_title('UMAP')

# 保存图像
plt.savefig("aa.png")

