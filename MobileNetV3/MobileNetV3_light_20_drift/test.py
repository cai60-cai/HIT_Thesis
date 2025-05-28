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
import torch.nn as nn

from dense_res_opt_mobilenetv3 import TrafficMobileNetV3

class FeatureAttention(nn.Module):
    """特征注意力机制"""
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        weights = self.attention(x)
        return x * weights

class TestDataset(Dataset):
    """测试数据集加载器"""
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

def save_class_metrics(metrics_dict, class_names, save_path):
    """Save per-class metrics to a CSV file."""
    filtered_metrics_dict = {
        key: metrics_dict[key] for key in class_names if key in metrics_dict
    }

    missing_classes = set(class_names) - set(filtered_metrics_dict.keys())
    for missing_class in missing_classes:
        filtered_metrics_dict[missing_class] = {
            "precision": 0.0,
            "recall": 0.0,
            "f1-score": 0.0,
            "support": 0,
        }

    df = pd.DataFrame(filtered_metrics_dict).T
    df.index = class_names
    df.to_csv(save_path)

    plt.figure(figsize=(15, 6))
    df[['precision', 'recall', 'f1-score']].plot(kind='bar')
    plt.title('Performance Metrics by Class')
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    plt.savefig(save_path.replace('.csv', '_plot.png'))
    plt.close()

def predict():
    """预测和评估模型性能"""
    # 配置
    data_dir = 'processed_data_drift'
    model_path = 'incremental_training_results_yuanshibujiatezhengceng/best_model.pth'
    output_dir = 'incremental_test_results_yuanshibujiatezhengceng'
    batch_size = 128
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载标签映射
    label_mapping = pd.read_csv(f"{data_dir}/label_mapping.csv", index_col=0)
    class_names = label_mapping.index.tolist()
    num_classes = len(class_names)
    
    # 加载模型
    model = TrafficMobileNetV3(num_classes=num_classes).to(device)
    
    # # 添加特征注意力层
    # in_features = model.classifier[0].in_features
    # model.classifier = nn.Sequential(
    #     FeatureAttention(in_features),
    #     model.classifier
    # )
    model = model.to(device)
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载测试数据
    test_dataset = TestDataset(data_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 预测
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            
            preds = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, os.path.join(output_dir, 'confusion_matrix.png'))
    
    # 保存混淆矩阵数据
    np.save(os.path.join(output_dir, 'confusion_matrix.npy'), cm)
    
    # 计算并保存每个类别的指标
    class_report = classification_report(all_labels, all_preds, 
                                      target_names=class_names, 
                                      output_dict=True)
    save_class_metrics(class_report, class_names, 
                      os.path.join(output_dir, 'class_metrics.csv'))
    
    # 保存预测概率
    np.save(os.path.join(output_dir, 'prediction_probabilities.npy'), 
            np.array(all_probs))
    
    # 保存每个样本的预测结果
    predictions_df = pd.DataFrame({
        'True_Label': [class_names[i] for i in all_labels],
        'Predicted_Label': [class_names[i] for i in all_preds],
        'Correct': np.array(all_labels) == np.array(all_preds)
    })
    predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), 
                         index=False)
    
    # 保存总体指标
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'model_info': {
            'epoch': checkpoint.get('epoch', 0),
            'best_val_metrics': checkpoint.get('metrics', {})
        }
    }
    
    with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # 打印结果
    print("\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # 分析错误预测
    error_analysis = predictions_df[~predictions_df['Correct']]
    error_counts = error_analysis.groupby(['True_Label', 'Predicted_Label']).size()
    
    print("\nTop Misclassifications:")
    print(error_counts.sort_values(ascending=False).head(10))
    
    # 保存错误分析
    error_counts.to_csv(os.path.join(output_dir, 'error_analysis.csv'))

if __name__ == '__main__':
    predict()