import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
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
from train import TrafficDataset
from enhanced_model import IncrementalTrafficMobileNetV3

class IncrementalTestDataset(Dataset):
    """增量测试数据集加载器"""
    def __init__(self, data_dir, prefix, is_new_class=False, label_offset=0):
        super().__init__()
        self.data_dir = data_dir
        self.prefix = prefix
        
        # 加载所有批次特征
        self.features = []
        batch_idx = 0
        while True:
            batch_path = f"{data_dir}/{prefix}_features_batch_{batch_idx}.npy"
            if not os.path.exists(batch_path):
                break
            self.features.append(np.load(batch_path))
            batch_idx += 1
        
        self.features = np.concatenate(self.features, axis=0)
        self.labels = np.load(f"{data_dir}/{prefix}_labels.npy")
        
        # 如果是新类别，添加标签偏移
        if is_new_class:
            self.labels = self.labels + label_offset
        
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

def predict_13_classes():
    """预测和评估13类模型性能"""
    # 配置
    original_data_dir = 'processed_data_org'  # 原始11类数据
    first_increment_dir = 'processed_data_incremental_last_5_3'  # 第12类数据
    second_increment_dir = 'processed_data_incremental_last_5_4'  # 第13类数据
    dir_list=[1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
    for dir in dir_list:
        model_path = f'training_results_incremental_13_port_scan_weights/port_scan_weight_{dir}/best_model.pth'  # 13类模型路径
        output_dir = f'test_results_13_classes_{dir}'
        batch_size = 128
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置设备
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # 定义类别名称
        class_names = [
            'BENIGN', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest',
            'DoS slowloris', 'FTP-Patator', 'Heartbleed', 'Infiltration', 'PortScan',
            'SSH-Patator', 'Web Attack', 'NewClass'
        ]
        
        # 创建模型实例
        model = IncrementalTrafficMobileNetV3(
            old_model_path='training_results_incremental_yuohua_webmokuai/best_model.pth',
            old_num_classes=12,
            new_num_classes=13
        ).to(device)
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 加载测试数据
        original_test = TrafficDataset(original_data_dir, 'test')
        first_increment_test = IncrementalTestDataset(
            first_increment_dir, 'test', is_new_class=True, label_offset=11
        )
        second_increment_test = IncrementalTestDataset(
            second_increment_dir, 'test', is_new_class=True, label_offset=12
        )
        
        # 合并数据集
        test_dataset = ConcatDataset([
            original_test, 
            first_increment_test, 
            second_increment_test
        ])
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # ... [其余代码保持不变] ...

        # [后续代码保持不变]
        
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
        
        # 计算总体指标
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # 生成混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        plot_confusion_matrix(cm, class_names, os.path.join(output_dir, 'confusion_matrix.png'))
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
        
        # 分别计算原始类别、第一次增量和第二次增量的性能
        def calculate_performance(mask):
            return {
                'accuracy': accuracy_score(
                    np.array(all_labels)[mask], np.array(all_preds)[mask]
                ),
                'precision': precision_score(
                    np.array(all_labels)[mask], np.array(all_preds)[mask], 
                    average='weighted'
                ),
                'recall': recall_score(
                    np.array(all_labels)[mask], np.array(all_preds)[mask], 
                    average='weighted'
                ),
                'f1': f1_score(
                    np.array(all_labels)[mask], np.array(all_preds)[mask], 
                    average='weighted'
                )
            }
        
        original_mask = np.array(all_labels) < 11
        first_increment_mask = np.array(all_labels) == 11
        second_increment_mask = np.array(all_labels) == 12
        
        metrics = {
            'overall': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            },
            'original_classes': calculate_performance(original_mask),
            'first_increment': calculate_performance(first_increment_mask),
            'second_increment': calculate_performance(second_increment_mask),
            'model_info': {
                'epoch': checkpoint.get('epoch', 0),
                'best_val_metrics': checkpoint.get('metrics', {})
            }
        }
        
        # 保存指标
        with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # 打印结果
        print("\nOverall Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        print("\nOriginal Classes Performance:")
        for metric, value in metrics['original_classes'].items():
            print(f"{metric}: {value:.4f}")
        
        print("\nFirst Increment Performance:")
        for metric, value in metrics['first_increment'].items():
            print(f"{metric}: {value:.4f}")
        
        print("\nSecond Increment Performance:")
        for metric, value in metrics['second_increment'].items():
            print(f"{metric}: {value:.4f}")
        
        # 分析错误预测
        error_analysis = predictions_df[~predictions_df['Correct']]
        error_counts = error_analysis.groupby(['True_Label', 'Predicted_Label']).size()
        print("\nTop Misclassifications:")
        print(error_counts.sort_values(ascending=False).head(10))
        error_counts.to_csv(os.path.join(output_dir, 'error_analysis.csv'))

if __name__ == '__main__':
    predict_13_classes()