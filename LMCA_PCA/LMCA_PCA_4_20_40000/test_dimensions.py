
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import time
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                           accuracy_score, confusion_matrix)
from LMCA import LMCA

class TestDataset(Dataset):
    """测试数据集加载器"""
    def __init__(self, data_dir):
        # 加载所有批次
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
        
        assert len(self.features) == len(self.labels), "特征和标签数量不匹配"
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx]).unsqueeze(0)
        label = torch.LongTensor([self.labels[idx]])[0]
        return feature, label

def calculate_score(metrics, model_size, inference_time, weights=None):
    """计算综合评分"""
    if weights is None:
        weights = {
            'accuracy': 0.35,
            'f1': 0.35,
            'model_size': 0.15,
            'inference_time': 0.15
        }
    
    # 归一化模型大小
    size_score = np.clip(1 - (model_size - 500000) / 1000000, 0, 1)
    
    # 归一化推理时间 (假设理想时间为10秒)
    time_score = np.clip(1 - (inference_time - 10) / 100, 0, 1)
    
    # 计算综合得分
    score = (
        weights['accuracy'] * metrics['accuracy'] +
        weights['f1'] * metrics['f1'] +
        weights['model_size'] * size_score +
        weights['inference_time'] * time_score
    ) * 100
    
    return score

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

def test_dimension(dim, base_dir='pca', batch_size=256):
    """测试特定维度的模型性能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建保存目录
    save_dir = os.path.join(base_dir, f'dim_{dim}', 'test_results')
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    data_dir = os.path.join(base_dir, f'dim_{dim}', 'processed_data')
    test_dataset = TestDataset(data_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 加载标签映射
    label_mapping = pd.read_csv(os.path.join(data_dir, 'label_mapping.csv'), index_col=0)
    class_names = label_mapping.index.tolist()
    
    # 加载模型
    matrix_size = int(np.ceil(np.sqrt(dim)))
    model = LMCA(input_dim=matrix_size, num_classes=13).to(device)
    
    # 加载最佳模型权重
    model_path = os.path.join(base_dir, f'dim_{dim}', 'training_results', 'best_model.pth')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 测试模式
    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()
    
    # 记录推理时间
    start_time = time.time()
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc=f'Testing dimension {dim}'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            preds = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(target.cpu().numpy())
    
    inference_time = time.time() - start_time
    
    # 计算指标
    test_metrics = {
        'loss': test_loss / len(test_loader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted'),
        'recall': recall_score(all_labels, all_preds, average='weighted'),
        'f1': f1_score(all_labels, all_preds, average='weighted')
    }
    
    # 计算并保存混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, os.path.join(save_dir, 'confusion_matrix.png'))
    np.save(os.path.join(save_dir, 'confusion_matrix.npy'), cm)
    
    # 计算模型大小
    model_size = sum(p.numel() for p in model.parameters())
    
    # 计算综合评分
    final_score = calculate_score(test_metrics, model_size, inference_time)
    
    # 保存测试结果
    results = {
        'dimension': dim,
        'matrix_size': matrix_size,
        'inference_time': inference_time,
        'model_size': model_size,
        'metrics': test_metrics,
        'score': final_score
    }
    
    # 保存为JSON
    with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # 保存每个类别的详细指标
    class_report = pd.DataFrame({
        'Precision': precision_score(all_labels, all_preds, average=None),
        'Recall': recall_score(all_labels, all_preds, average=None),
        'F1': f1_score(all_labels, all_preds, average=None)
    }, index=class_names)
    
    class_report.to_csv(os.path.join(save_dir, 'class_report.csv'))
    
    return results

def plot_dimension_comparison(all_results, base_dir='pca'):
    """绘制不同维度的测试结果比较"""
    dimensions = [r['dimension'] for r in all_results]
    scores = [r['score'] for r in all_results]
    accuracies = [r['metrics']['accuracy'] for r in all_results]
    f1_scores = [r['metrics']['f1'] for r in all_results]
    times = [r['inference_time'] for r in all_results]
    
    plt.figure(figsize=(15, 10))
    
    # 综合评分
    plt.subplot(2, 2, 1)
    plt.plot(dimensions, scores, 'o-')
    plt.title('Dimension vs Overall Score')
    plt.xlabel('Dimension')
    plt.ylabel('Score')
    plt.grid(True)
    
    # 准确率
    plt.subplot(2, 2, 2)
    plt.plot(dimensions, accuracies, 'o-')
    plt.title('Dimension vs Accuracy')
    plt.xlabel('Dimension')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # F1分数
    plt.subplot(2, 2, 3)
    plt.plot(dimensions, f1_scores, 'o-')
    plt.title('Dimension vs F1 Score')
    plt.xlabel('Dimension')
    plt.ylabel('F1 Score')
    plt.grid(True)
    
    # 推理时间
    plt.subplot(2, 2, 4)
    plt.plot(dimensions, times, 'o-')
    plt.title('Dimension vs Inference Time')
    plt.xlabel('Dimension')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'test_comparison.png'))
    plt.close()

def main():
    base_dir = 'pca'
    dimensions = [10, 20, 30, 40, 50, 60, 70, 78]
    
    all_results = []
    
    # 测试每个维度
    for dim in dimensions:
        print(f"\nTesting dimension {dim}")
        results = test_dimension(dim, base_dir)
        all_results.append(results)
        
        # 保存当前进度
        pd.DataFrame(all_results).to_csv(
            os.path.join(base_dir, 'test_results.csv'), 
            index=False
        )
    
    # 绘制比较图
    plot_dimension_comparison(all_results, base_dir)
    
    # 找出最佳维度
    best_result = max(all_results, key=lambda x: x['score'])
    print("\nBest dimension in testing:")
    print(f"Dimension: {best_result['dimension']}")
    print(f"Score: {best_result['score']:.2f}")
    print(f"Accuracy: {best_result['metrics']['accuracy']:.4f}")
    print(f"F1 Score: {best_result['metrics']['f1']:.4f}")
    print(f"Inference Time: {best_result['inference_time']:.2f} seconds")

if __name__ == '__main__':
    main()
