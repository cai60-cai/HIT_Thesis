
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

class NetworkTrafficDataset(Dataset):
    """网络流量数据集加载器"""
    def __init__(self, data_dir, prefix):
        self.data_dir = data_dir
        self.prefix = prefix
        
        # 加载所有批次的特征
        self.features = []
        batch_idx = 0
        while True:
            batch_path = f"{data_dir}/{prefix}_features_batch_{batch_idx}.npy"
            if not os.path.exists(batch_path):
                break
            self.features.append(np.load(batch_path))
            batch_idx += 1
        
        # 合并所有批次
        self.features = np.concatenate(self.features, axis=0)
        
        # 加载标签
        self.labels = np.load(f"{data_dir}/{prefix}_labels.npy")
        
        # 确保数据和标签长度匹配
        assert len(self.features) == len(self.labels), "特征和标签数量不匹配"
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx]).unsqueeze(0)  # 添加通道维度
        label = torch.LongTensor([self.labels[idx]])[0]
        return feature, label

def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 收集预测结果
        preds = output.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(target.cpu().numpy())
        
        # 更新进度条
        pbar.set_postfix({'loss': f'{total_loss/(batch_idx+1):.4f}'})
    
    # 计算指标
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return total_loss / len(dataloader), precision, recall, accuracy, f1

def validate(model, dataloader, criterion, device):
    """验证函数"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc='Validation'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            
            preds = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(target.cpu().numpy())
    
    # 计算指标
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return total_loss / len(dataloader), precision, recall, accuracy, f1

def plot_metrics(metrics_history, save_dir):
    """绘制训练指标变化曲线"""
    epochs = [m['epoch'] for m in metrics_history]
    
    # 创建图形
    plt.figure(figsize=(15, 10))
    
    # 绘制损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(epochs, [m['train_loss'] for m in metrics_history], label='Train Loss')
    plt.plot(epochs, [m['val_loss'] for m in metrics_history], label='Val Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(epochs, [m['train_accuracy'] for m in metrics_history], label='Train Accuracy')
    plt.plot(epochs, [m['val_accuracy'] for m in metrics_history], label='Val Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 绘制F1分数曲线
    plt.subplot(2, 2, 3)
    plt.plot(epochs, [m['train_f1'] for m in metrics_history], label='Train F1')
    plt.plot(epochs, [m['val_f1'] for m in metrics_history], label='Val F1')
    plt.title('F1 Score vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    
    # 绘制Precision和Recall
    plt.subplot(2, 2, 4)
    plt.plot(epochs, [m['train_precision'] for m in metrics_history], label='Train Precision')
    plt.plot(epochs, [m['train_recall'] for m in metrics_history], label='Train Recall')
    plt.plot(epochs, [m['val_precision'] for m in metrics_history], label='Val Precision')
    plt.plot(epochs, [m['val_recall'] for m in metrics_history], label='Val Recall')
    plt.title('Precision & Recall vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()
