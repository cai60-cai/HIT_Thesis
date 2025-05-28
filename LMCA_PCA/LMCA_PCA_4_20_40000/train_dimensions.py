
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import json
import time
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from LMCA import LMCA

class DimensionDataset(Dataset):
    """数据集加载器"""
    def __init__(self, data_dir, prefix):
        self.data_dir = data_dir
        self.prefix = prefix
        
        # 加载所有批次
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
        
        assert len(self.features) == len(self.labels), "特征和标签数量不匹配"
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx]).unsqueeze(0)
        label = torch.LongTensor([self.labels[idx]])[0]
        return feature, label

def calculate_score(metrics, model_size, train_time, weights=None):
    """计算综合评分
    
    Args:
        metrics: 包含acc, f1等指标的字典
        model_size: 模型参数量
        train_time: 训练时间(秒)
        weights: 各指标的权重字典
    
    Returns:
        float: 综合评分 (0-100)
    """
    if weights is None:
        weights = {
            'accuracy': 0.35,
            'f1': 0.35,
            'model_size': 0.15,
            'train_time': 0.15
        }
    
    # 归一化模型大小 (假设理想大小为500K参数)
    size_score = np.clip(1 - (model_size - 500000) / 1000000, 0, 1)
    
    # 归一化训练时间 (假设理想时间为1小时)
    time_score = np.clip(1 - (train_time - 3600) / 7200, 0, 1)
    
    # 计算综合得分
    score = (
        weights['accuracy'] * metrics['accuracy'] +
        weights['f1'] * metrics['f1'] +
        weights['model_size'] * size_score +
        weights['train_time'] * time_score
    ) * 100
    
    return score

def train_dimension(dim, base_dir='pca', num_epochs=20, batch_size=256):
    """训练特定维度的模型"""
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # 创建保存目录
    save_dir = os.path.join(base_dir, f'dim_{dim}', 'training_results')
    epoch_dir = os.path.join(save_dir, 'epochs')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(epoch_dir, exist_ok=True)
    
    # 加载数据
    data_dir = os.path.join(base_dir, f'dim_{dim}', 'processed_data')
    dataset = DimensionDataset(data_dir, 'train')
    
    # 划分训练集和验证集 (95:5)
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 初始化模型
    matrix_size = int(np.ceil(np.sqrt(dim)))
    model = LMCA(input_dim=matrix_size, num_classes=13).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # 训练配置
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 记录训练历史
    history = []
    best_val_f1 = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        
        # 为当前epoch创建保存目录
        current_epoch_dir = os.path.join(epoch_dir, f'epoch_{epoch+1}')
        os.makedirs(current_epoch_dir, exist_ok=True)
        # 训练
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for data, target in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = output.argmax(dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(target.cpu().numpy())
        
        # 验证
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                preds = output.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(target.cpu().numpy())
        
        # 计算指标
        train_metrics = {
            'loss': train_loss / len(train_loader),
            'accuracy': accuracy_score(train_labels, train_preds),
            'precision': precision_score(train_labels, train_preds, average='weighted'),
            'recall': recall_score(train_labels, train_preds, average='weighted'),
            'f1': f1_score(train_labels, train_preds, average='weighted')
        }
        
        val_metrics = {
            'loss': val_loss / len(val_loader),
            'accuracy': accuracy_score(val_labels, val_preds),
            'precision': precision_score(val_labels, val_preds, average='weighted'),
            'recall': recall_score(val_labels, val_preds, average='weighted'),
            'f1': f1_score(val_labels, val_preds, average='weighted')
        }
        
        # 学习率调整
        scheduler.step()
        
        # 保存当前epoch的数据
        epoch_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        
        # 保存checkpoint
        torch.save(epoch_data, os.path.join(current_epoch_dir, 'checkpoint.pth'))
        
        # 保存指标
        metrics_data = {
            'train': train_metrics,
            'val': val_metrics,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        with open(os.path.join(current_epoch_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_data, f, indent=4)
        
        # 保存混淆矩阵
        plot_confusion_matrix(
            val_labels, 
            val_preds,
            save_path=os.path.join(current_epoch_dir, 'confusion_matrix.png')
        )

        # 记录历史
        history.append({
            'epoch': epoch + 1,
            'train': train_metrics,
            'val': val_metrics,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # 如果是最佳模型则保存
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            # 保存最佳模型
            torch.save(epoch_data, os.path.join(save_dir, 'best_model.pth'))
            # 记录最佳epoch
            with open(os.path.join(save_dir, 'best_epoch.txt'), 'w') as f:
                f.write(f"Best model from epoch {epoch+1}")
        
        # 打印当前epoch结果
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
    
    # 计算训练时间和模型大小
    train_time = time.time() - start_time
    model_size = sum(p.numel() for p in model.parameters())
    
    # 计算综合评分
    final_score = calculate_score(val_metrics, model_size, train_time)
    
    # 保存训练完成后的汇总信息
    summary = {
        'dimension': dim,
        'matrix_size': matrix_size,
        'train_time': train_time,
        'model_size': model_size,
        'best_epoch': max(range(num_epochs), key=lambda e: history[e]['val']['f1']) + 1,
        'best_val_f1': best_val_f1,
        'final_metrics': val_metrics,
        'score': final_score,
        'architecture': {
            'input_dim': matrix_size,
            'num_classes': 11,
            'total_params': model_size
        },
        'training_config': {
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'optimizer': 'Adam',
            'initial_lr': 1e-4,
            'weight_decay': 5e-4
        }
    }
    with open(os.path.join(save_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    # 保存训练历史
    pd.DataFrame(history).to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)
    
    # 绘制训练曲线
    plot_training_curves(history, save_dir)
    return summary
    # # 保存训练结果
    # results = {
    #     'dimension': dim,
    #     'matrix_size': matrix_size,
    #     'train_time': train_time,
    #     'model_size': model_size,
    #     'best_val_f1': best_val_f1,
    #     'final_metrics': val_metrics,
    #     'score': final_score
    # }
    
    # # 保存历史记录
    # pd.DataFrame(history).to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)
    
    # # 保存结果摘要
    # with open(os.path.join(save_dir, 'training_results.json'), 'w') as f:
    #     json.dump(results, f, indent=4)
    
    # return results


def plot_training_curves(history, save_dir):
    """绘制详细的训练曲线"""
    metrics = ['loss', 'accuracy', 'f1', 'precision', 'recall']
    plt.figure(figsize=(20, 15))
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(3, 2, i)
        plt.plot([h['train'][metric] for h in history], label=f'Train {metric}')
        plt.plot([h['val'][metric] for h in history], label=f'Val {metric}')
        plt.title(f'{metric.capitalize()} vs Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
    
    # 学习率曲线
    plt.subplot(3, 2, 6)
    plt.plot([h['lr'] for h in history])
    plt.title('Learning Rate vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path):
    """绘制混淆矩阵"""
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_dimension_comparison(all_results, base_dir='pca'):
    """绘制不同维度的比较图"""
    dimensions = [r['dimension'] for r in all_results]
    scores = [r['score'] for r in all_results]
    accuracies = [r['final_metrics']['accuracy'] for r in all_results]
    f1_scores = [r['final_metrics']['f1'] for r in all_results]
    train_times = [r['train_time'] / 3600 for r in all_results]  # 转换为小时
    
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
    
    # 训练时间
    plt.subplot(2, 2, 4)
    plt.plot(dimensions, train_times, 'o-')
    plt.title('Dimension vs Training Time')
    plt.xlabel('Dimension')
    plt.ylabel('Training Time (hours)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'dimension_comparison.png'))
    plt.close()

def main():
    # 配置
    base_dir = 'pca'
    dimensions = [10, 20, 30, 40, 50, 60, 70, 78]
    
    all_results = []
    
    # 训练每个维度
    for dim in dimensions:
        print(f"\nTraining dimension {dim}")
        results = train_dimension(dim, base_dir)
        all_results.append(results)
        
        # 保存当前进度
        pd.DataFrame(all_results).to_csv(
            os.path.join(base_dir, 'dimension_results.csv'), 
            index=False
        )
    
    # 绘制比较图
    plot_dimension_comparison(all_results, base_dir)
    
    # 找出最佳维度
    best_result = max(all_results, key=lambda x: x['score'])
    print("\nBest dimension found:")
    print(f"Dimension: {best_result['dimension']}")
    print(f"Score: {best_result['score']:.2f}")
    print(f"Accuracy: {best_result['final_metrics']['accuracy']:.4f}")
    print(f"F1 Score: {best_result['final_metrics']['f1']:.4f}")
    print(f"Training Time: {best_result['train_time']/3600:.2f} hours")

if __name__ == '__main__':
    main()