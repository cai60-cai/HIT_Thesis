import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, 
                           f1_score, accuracy_score, classification_report)
from tqdm import tqdm
from dense_res_opt_mobilenetv3 import TrafficMobileNetV3

class BaselineTestDataset(Dataset):
    """用于基准测试的数据加载器"""
    def __init__(self, data_dir):
        # 加载测试数据
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

def plot_metrics_comparison(metrics_dict, save_path):
    """绘制性能指标对比图"""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    values = [metrics_dict[m] for m in metrics]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values)
    plt.title('Baseline Model Performance Metrics')
    plt.ylabel('Score')
    
    # 在柱状图上添加具体数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.ylim(0, 1.1)  # 设置y轴范围
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
def plot_confusion_matrix(cm, class_names, save_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def calculate_metrics(y_true, y_pred, class_names):
    """重新计算每个类别的指标"""
    metrics = {}
    
    # 为每个类别计算指标
    for idx, class_name in enumerate(class_names):
        # 该类的二分类问题
        class_mask = y_true == idx
        pred_mask = y_pred == idx
        
        # 计算该类的accuracy (正确预测的样本占总样本的比例)
        class_accuracy = np.mean(pred_mask == class_mask)
        
        # 计算TP, FP, FN
        tp = np.sum((y_true == idx) & (y_pred == idx))
        fp = np.sum((y_true != idx) & (y_pred == idx))
        fn = np.sum((y_true == idx) & (y_pred != idx))
        
        # 计算其他指标
        support = np.sum(class_mask)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[class_name] = {
            'accuracy': class_accuracy,  # 该类的准确率
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': int(support)
        }
    
    # 计算平均值
    accuracies = [m['accuracy'] for m in metrics.values()]
    precisions = [m['precision'] for m in metrics.values()]
    recalls = [m['recall'] for m in metrics.values()]
    f1_scores = [m['f1-score'] for m in metrics.values()]
    supports = [m['support'] for m in metrics.values()]
    
    # Macro averages
    metrics['macro avg'] = {
        'accuracy': np.mean(accuracies),
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1-score': np.mean(f1_scores),
        'support': sum(supports)
    }
    
    # Weighted averages
    weights = np.array(supports) / sum(supports)
    metrics['weighted avg'] = {
        'accuracy': np.average(accuracies, weights=weights),
        'precision': np.average(precisions, weights=weights),
        'recall': np.average(recalls, weights=weights),
        'f1-score': np.average(f1_scores, weights=weights),
        'support': sum(supports)
    }
    
    return metrics

def save_metrics_csv(metrics, save_path):
    """保存指标为CSV格式"""
    rows = []
    columns = ['accuracy', 'precision', 'recall', 'f1-score', 'support']
    
    for class_name, class_metrics in metrics.items():
        row = {col: class_metrics[col] for col in columns}
        rows.append({
            'Class': class_name,
            **row
        })
    
    df = pd.DataFrame(rows)
    df.set_index('Class', inplace=True)
    
    # 格式化数值
    for col in columns[:-1]:  # 除了support
        df[col] = df[col].apply(lambda x: f"{x:.4f}")
    df['support'] = df['support'].apply(lambda x: f"{int(float(x))}")
    
    df.to_csv(save_path)
    return df

def load_partial_state_dict(model, checkpoint):
    model_state = model.state_dict()
    loaded_state = checkpoint['model_state_dict']
    
    # 筛选出兼容的层
    compatible_state = {k: v for k, v in loaded_state.items() 
                        if k in model_state and v.shape == model_state[k].shape}
    
    model_state.update(compatible_state)
    model.load_state_dict(model_state)
    return model


def evaluate_baseline():
    """评估模型性能"""
    # 配置
    data_dir = 'processed_data_incremental'
    model_path = 'model/best_model.pth'
    output_dir = 'baseline_evaluation_results_incremental'
    batch_size = 256
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据和模型
    label_mapping = pd.read_csv(f"{data_dir}/label_mapping.csv", index_col=0)
    class_names = label_mapping.index.tolist()
    
    # 在 evaluate_baseline 函数中：   44
    model = TrafficMobileNetV3(num_classes=len(class_names)).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model = load_partial_state_dict(model, checkpoint)

#   ##     1111111111111
#     model = TrafficMobileNetV3(num_classes=len(class_names)).to(device)
#     checkpoint = torch.load(model_path, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
    
    # 评估过程
    test_dataset = BaselineTestDataset(data_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算指标
    metrics = calculate_metrics(all_labels, all_preds, class_names)
    
    # 保存结果
    save_metrics_csv(metrics, os.path.join(output_dir, 'classification_metrics.csv'))
    
    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    np.save(os.path.join(output_dir, 'confusion_matrix.npy'), cm)
    plot_confusion_matrix(cm, class_names, os.path.join(output_dir, 'confusion_matrix.png'))
    
    # 打印结果
    print("\nEvaluation Results:")
    print("=" * 40)
    for class_name, class_metrics in metrics.items():
        if class_name not in ['macro avg', 'weighted avg']:
            print(f"\n{class_name}:")
            for metric, value in class_metrics.items():
                if metric == 'support':
                    print(f"{metric}: {int(value)}")
                else:
                    print(f"{metric}: {value:.4f}")


if __name__ == '__main__':
    evaluate_baseline()