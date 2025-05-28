import json
import matplotlib.pyplot as plt
import os
import re

def extract_loss(json_string):
    # 使用正则表达式提取loss值
    match = re.search(r"'loss': ([\d.]+)", json_string)
    if match:
        return float(match.group(1))
    return None

def plot_loss_from_json(input_dir):
    # 获取所有维度并分成两组
    dimensions = [10, 20, 30, 40, 50, 60, 70, 78]
    dimension_groups = [dimensions[:4], dimensions[4:]]
    
    for group_idx, dims in enumerate(dimension_groups):
        # 创建2x2的子图布局
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes_flat = axes.flatten()
        
        max_loss = 0  # 用于统一y轴范围
        
        # 首先找到最大loss值
        for dim in dims:
            json_path = os.path.join(input_dir, f'dim_{dim}_train.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    loss_values = [extract_loss(item) for item in data if extract_loss(item) is not None]
                    if loss_values:
                        max_loss = max(max_loss, max(loss_values))
        
        # 绘制每个维度的图
        for idx, dim in enumerate(dims):
            json_path = os.path.join(input_dir, f'dim_{dim}_train.json')
            
            if os.path.exists(json_path):
                print(f"Processing dimension {dim}")
                
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                    # 提取loss值
                    loss_values = []
                    for item in data:
                        loss = extract_loss(item)
                        if loss is not None:
                            loss_values.append(loss)
                    
                    if loss_values:
                        # 绘制子图
                        ax = axes_flat[idx]
                        epochs = range(1, len(loss_values) + 1)
                        ax.plot(epochs, loss_values, marker='o', markersize=2, color='blue')
                        ax.set_title(f'Dimension {dim}')
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Loss')
                        ax.grid(True)
                        
                        # 统一y轴范围
                        ax.set_ylim(0, max_loss * 1.1)
        
        # 调整子图之间的间距
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(f'pca_dimensions_loss_group_{group_idx+1}.png', dpi=300, bbox_inches='tight')
        plt.close()

# 使用示例
input_directory = 'plt'  # JSON文件所在的目录
plot_loss_from_json(input_directory)