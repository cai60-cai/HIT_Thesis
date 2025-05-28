import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import sys
from datetime import datetime
from dense_res_opt_mobilenetv3 import TrafficMobileNetV3

class NewClassTestDataset(Dataset):
    """新类别测试数据集加载器"""
    def __init__(self, data_dir):
        self.features = []
        batch_idx = 0
        while True:
            batch_path = f"{data_dir}/test_features_batch_{batch_idx}.npy"
            if not os.path.exists(batch_path):
                break
            self.features.append(np.load(batch_path))
            batch_idx += 1
            
        self.features = np.concatenate(self.features, axis=0)
        self.labels = np.load(f"{data_dir}/test_labels.npy")
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx]).unsqueeze(0)
        label = torch.LongTensor([self.labels[idx]])[0]
        return feature, label

def plot_distribution(probs, true_labels, old_class_names, new_class_names, save_path, new_label_mapping):
    """改进的预测概率分布绘制函数"""
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # 添加调试信息
    print("Starting plot_distribution...")
    print(f"Number of probabilities: {len(probs)}")
    print(f"Number of true labels: {len(true_labels)}")
    print(f"Old class names: {old_class_names}")
    print(f"New class names: {new_class_names}")

    sns.set(style='darkgrid')  # 使用 seaborn 的样式设置
    num_plots = len(new_class_names)
    cols = 2
    rows = (num_plots + 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axs = axs.ravel()

    for idx, new_class in enumerate(new_class_names):
        if idx >= len(axs):
            print(f"Warning: More new classes ({len(new_class_names)}) than subplots ({len(axs)}).")
            break

        print(f"Plotting for new class: {new_class}")
        # 获取该类别的所有样本的预测概率
        try:
            class_idx = new_label_mapping.loc[new_class].iloc[0]
        except Exception as e:
            print(f"Error retrieving class index for {new_class}: {e}")
            continue

        class_mask = np.array(true_labels) == class_idx
        class_probs = np.array(probs)[class_mask]

        if len(class_probs) > 0:
            # 计算平均预测概率
            mean_probs = class_probs.mean(axis=0)
            print(f"Mean probabilities for {new_class}: {mean_probs}")

            # 绘制条形图
            ax = axs[idx]
            bars = ax.bar(range(len(old_class_names)), mean_probs, color='skyblue', edgecolor='black')

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                if height > 0:  # 只显示非零值
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.4f}',
                            ha='center', va='bottom', rotation=0)

            # 设置坐标轴
            ax.set_xticks(range(len(old_class_names)))
            ax.set_xticklabels(old_class_names, rotation=90, ha='right')
            ax.set_title(f'Average Prediction Distribution for {new_class}', pad=20)
            ax.set_ylabel('Average Probability')

            # 设置y轴范围
            max_prob = max(mean_probs)
            ax.set_ylim(0, max(0.05, max_prob * 1.2))  # 确保小值也能显示

            # 添加网格线
            ax.grid(True, linestyle='--', alpha=0.7)

            # 突出显示最高概率的类别
            max_idx = np.argmax(mean_probs)
            bars[max_idx].set_color('red')
        else:
            print(f"No data found for class: {new_class}")

    # 移除多余的子图
    for j in range(idx + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.suptitle('Prediction Distribution for New Classes', y=1.02, fontsize=16)
    plt.tight_layout()

    # 添加调试信息
    print(f"Saving plot to: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Plot saved successfully.")



def analyze_new_classes():
    """分析新类别的预测情况"""
    # 配置
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'incremental_class_analysis_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志文件
    log_file = os.path.join(output_dir, 'analysis_log.txt')
    sys.stdout = open(log_file, 'w', encoding='utf-8')
    
    # 配置参数
    data_dir = 'processed_data_incremental_2'
    model_path = 'model/best_model.pth'
    batch_size = 256
    
    # 设备设置
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"=== Analysis Started at {timestamp} ===")
    print(f"Using device: {device}")
    
    # 加载标签映射
    old_label_mapping = pd.read_csv("processed_data_org/label_mapping.csv", index_col=0)
    new_label_mapping = pd.read_csv(f"{data_dir}/label_mapping.csv", index_col=0)
    
    old_class_names = old_label_mapping.index.tolist()
    all_class_names = new_label_mapping.index.tolist()
    new_class_names = [c for c in all_class_names if c not in old_class_names]
    
    print("\nClass Information:")
    print(f"Original classes ({len(old_class_names)}): {old_class_names}")
    print(f"New classes ({len(new_class_names)}): {new_class_names}")
    
    # 加载模型
    model = TrafficMobileNetV3(num_classes=len(old_class_names)).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("\nModel loaded successfully")
    
    # 加载测试数据
    test_dataset = NewClassTestDataset(data_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 预测
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\nStarting prediction process...")
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing new classes"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            
            preds = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 分析结果
    print("\nAnalyzing predictions...")
    new_class_analysis = defaultdict(dict)
    
    for new_class in new_class_names:
        print(f"\n=== Analysis for {new_class} ===")
        class_idx = new_label_mapping.loc[new_class].iloc[0]
        class_mask = np.array(all_labels) == class_idx
        class_preds = np.array(all_preds)[class_mask]
        class_probs = np.array(all_probs)[class_mask]
        
        # 统计预测分布
        pred_distribution = pd.Series(class_preds).value_counts().sort_index()
        
        # 详细统计
        new_class_analysis[new_class] = {
            'total_samples': len(class_preds),
            'prediction_distribution': {old_class_names[i]: int(count) 
                                     for i, count in pred_distribution.items()},
            'average_confidence': class_probs.mean(axis=0).tolist(),
            'max_confidence': class_probs.max(axis=0).tolist(),
            'min_confidence': class_probs.min(axis=0).tolist(),
            'std_confidence': class_probs.std(axis=0).tolist()
        }
        
        # 打印详细分析
        print(f"Total samples: {len(class_preds)}")
        print("\nPrediction distribution:")
        
        # 按预测数量排序
        sorted_predictions = sorted(
            new_class_analysis[new_class]['prediction_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for old_class, count in sorted_predictions:
            percentage = (count / len(class_preds)) * 100
            confidence_idx = old_class_names.index(old_class)
            avg_conf = new_class_analysis[new_class]['average_confidence'][confidence_idx]
            print(f"{old_class}: {count} samples ({percentage:.2f}%) - Avg conf: {avg_conf:.4f}")
        
        # 置信度统计
        print("\nConfidence statistics:")
        class_max_conf_idx = np.argmax(new_class_analysis[new_class]['average_confidence'])
        print(f"Most confident prediction: {old_class_names[class_max_conf_idx]}")
        print(f"Average max confidence: {max(new_class_analysis[new_class]['average_confidence']):.4f}")
        print(f"Overall max confidence: {max(new_class_analysis[new_class]['max_confidence']):.4f}")
        print(f"Overall min confidence: {min(new_class_analysis[new_class]['min_confidence']):.4f}")
        print(f"Confidence std dev: {np.mean(new_class_analysis[new_class]['std_confidence']):.4f}")
    
    # 保存结果
    print("\nSaving results...")
    
    # 保存分析结果
    with open(os.path.join(output_dir, 'new_class_analysis.json'), 'w') as f:
        json.dump(new_class_analysis, f, indent=4)
    
    # 保存预测概率
    np.save(os.path.join(output_dir, 'prediction_probabilities.npy'), np.array(all_probs))
    
    # 保存原始预测和标签
    np.save(os.path.join(output_dir, 'predictions.npy'), np.array(all_preds))
    np.save(os.path.join(output_dir, 'true_labels.npy'), np.array(all_labels))
    
    # 绘制并保存分布图
    plot_distribution(all_probs, all_labels, old_class_names, new_class_names, 
                  os.path.join(output_dir, 'prediction_distribution.png'),
                  new_label_mapping)

    
    print(f"\nResults saved to {output_dir}")
    print(f"=== Analysis Completed at {datetime.now().strftime('%Y%m%d_%H%M%S')} ===")
    
    return new_class_analysis

if __name__ == '__main__':
    analyze_new_classes()
