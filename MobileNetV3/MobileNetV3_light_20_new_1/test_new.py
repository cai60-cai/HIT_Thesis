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
# from IncrementalMobileNetV3 import IncrementalTrafficMobileNetV3
from enhanced_model import IncrementalTrafficMobileNetV3

class IncrementalTestDataset(Dataset):
    """增量学习测试数据集加载器"""
    def __init__(self, old_data_dir, new_data_dir):
        # 加载旧类别测试数据
        self.old_features = []
        batch_idx = 0
        while True:
            batch_path = f"{old_data_dir}/test_features_batch_{batch_idx}.npy"
            if not os.path.exists(batch_path):
                break
            self.old_features.append(np.load(batch_path))
            batch_idx += 1
        
        # 加载新类别测试数据
        self.new_features = []
        batch_idx = 0
        while True:
            batch_path = f"{new_data_dir}/test_features_batch_{batch_idx}.npy"
            if not os.path.exists(batch_path):
                break
            self.new_features.append(np.load(batch_path))
            batch_idx += 1
            
        # 合并特征和标签
        self.old_features = np.concatenate(self.old_features, axis=0)
        self.new_features = np.concatenate(self.new_features, axis=0)
        self.features = np.concatenate([self.old_features, self.new_features], axis=0)
        
        # 加载并合并标签
        old_labels = np.load(f"{old_data_dir}/test_labels.npy")
        new_labels = np.load(f"{new_data_dir}/test_labels.npy")
        new_labels = new_labels + 11  # 调整新类别的标签
        self.labels = np.concatenate([old_labels, new_labels])
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx]).unsqueeze(0)
        label = torch.LongTensor([self.labels[idx]])[0]
        return feature, label

def plot_confusion_matrix(cm, class_names, save_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(15, 12))
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
    """保存每个类别的指标"""
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

def predict_incremental():
    """预测和评估增量学习模型性能"""
    # 配置
    old_data_dir = 'processed_data_org'
    new_data_dir = 'processed_data_incremental_last_5_3'  # 新类别数据目录
    model_path = 'training_results_incremental/best_model.pth'
    dir = "buyuohua_webmokuai"
    model_path = f'training_results_incremental_{dir}/best_model.pth'
    # model_path = 'training_results_incremental_yuohua_webmokuai/best_model.pth'
    output_dir = f'test_results_incremental_{dir}'
    batch_size = 128
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载标签映射并添加新类别
    old_label_mapping = pd.read_csv(f"{old_data_dir}/label_mapping.csv", index_col=0)
    class_names = old_label_mapping.index.tolist()
    class_names.append("NewClass")  # 添加新类别的名称
    num_classes = len(class_names)
    
    # 加载模型和检查点
    checkpoint = torch.load(model_path, map_location=device)
    model = IncrementalTrafficMobileNetV3(
        old_model_path='model/best_model.pth',  # 提供原始模型路径
        old_num_classes=11,
        new_num_classes=12
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载测试数据
    test_dataset = IncrementalTestDataset(old_data_dir, new_data_dir)
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
    
    # 分别计算旧类别和新类别的性能
    old_mask = np.array(all_labels) < 11
    new_mask = np.array(all_labels) >= 11
    
    # 旧类别性能
    old_metrics = {
        'accuracy': accuracy_score(np.array(all_labels)[old_mask], np.array(all_preds)[old_mask]),
        'precision': precision_score(np.array(all_labels)[old_mask], np.array(all_preds)[old_mask], average='weighted'),
        'recall': recall_score(np.array(all_labels)[old_mask], np.array(all_preds)[old_mask], average='weighted'),
        'f1_score': f1_score(np.array(all_labels)[old_mask], np.array(all_preds)[old_mask], average='weighted')
    }
    
    # 新类别性能
    new_metrics = {
        'accuracy': accuracy_score(np.array(all_labels)[new_mask], np.array(all_preds)[new_mask]),
        'precision': precision_score(np.array(all_labels)[new_mask], np.array(all_preds)[new_mask], average='weighted'),
        'recall': recall_score(np.array(all_labels)[new_mask], np.array(all_preds)[new_mask], average='weighted'),
        'f1_score': f1_score(np.array(all_labels)[new_mask], np.array(all_preds)[new_mask], average='weighted')
    }
    
    # 保存总体指标
    metrics = {
        'overall': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        },
        'old_classes': old_metrics,
        'new_class': new_metrics,
        'model_info': {
            'epoch': checkpoint.get('epoch', 0),
            'best_val_metrics': checkpoint.get('metrics', {})
        }
    }
    
    with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # 打印结果
    print("\nOverall Test Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print("\nOld Classes Performance:")
    print(f"Accuracy: {old_metrics['accuracy']:.4f}")
    print(f"F1 Score: {old_metrics['f1_score']:.4f}")
    
    print("\nNew Class Performance:")
    print(f"Accuracy: {new_metrics['accuracy']:.4f}")
    print(f"F1 Score: {new_metrics['f1_score']:.4f}")
    
    # 分析错误预测
    error_analysis = predictions_df[~predictions_df['Correct']]
    error_counts = error_analysis.groupby(['True_Label', 'Predicted_Label']).size()
    
    print("\nTop Misclassifications:")
    print(error_counts.sort_values(ascending=False).head(10))
    
    # 保存错误分析
    error_counts.to_csv(os.path.join(output_dir, 'error_analysis.csv'))

if __name__ == '__main__':
    predict_incremental()