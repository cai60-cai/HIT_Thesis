
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from dense_res_opt_mobilenetv3 import TrafficMobileNetV3

import pandas as pd

class TrafficDataset(Dataset):
    """数据集加载器"""
    def __init__(self, data_dir, prefix):
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
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx]).unsqueeze(0)
        label = torch.LongTensor([self.labels[idx]])[0]
        return feature, label

def plot_metrics(history, save_path):
    """绘制训练指标曲线"""
    plt.figure(figsize=(20, 12))
    
    # 损失曲线
    plt.subplot(2, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 准确率曲线
    plt.subplot(2, 3, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # F1分数曲线
    plt.subplot(2, 3, 3)
    plt.plot(history['train_f1'], label='Train F1')
    plt.plot(history['val_f1'], label='Val F1')
    plt.title('F1 Score vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    
    # 精确率曲线
    plt.subplot(2, 3, 4)
    plt.plot(history['train_precision'], label='Train Precision')
    plt.plot(history['val_precision'], label='Val Precision')
    plt.title('Precision vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    
    # 召回率曲线
    plt.subplot(2, 3, 5)
    plt.plot(history['train_recall'], label='Train Recall')
    plt.plot(history['val_recall'], label='Val Recall')
    plt.title('Recall vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)
    
    # 学习率曲线
    plt.subplot(2, 3, 6)
    plt.plot(history['learning_rate'], label='Learning Rate')
    plt.title('Learning Rate vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, class_names, save_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train():
    # 配置参数
    data_dir = 'processed_data_incremental_last'
    save_dir = 'training_results_incremental_last'
    num_epochs = 30
    batch_size = 256
    learning_rate = 1e-3
    weight_decay = 1e-5
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    dataset = TrafficDataset(data_dir, 'train')
    
    # 加载标签映射
    label_mapping = pd.read_csv(f"{data_dir}/label_mapping.csv", index_col=0)
    class_names = label_mapping.index.tolist()
    
    # 划分训练集和验证集 (95:5)
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 初始化模型
    # model = TrafficMobileNetV3(num_classes=len(class_names)).to(device)
    # model = MobileNetV3(num_classes=11).to(device)
    # model = TrafficMobileNetV3(num_classes=len(class_names)).to(device)
    model = TrafficMobileNetV3(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 训练历史
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [], 
        'train_precision': [], 'train_recall': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 
        'val_precision': [], 'val_recall': [],
        'learning_rate': []
    }
    
    best_val_f1 = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # 训练阶段
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc='Training')):
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
        
        train_metrics = {
            'loss': train_loss / len(train_loader),
            'accuracy': accuracy_score(train_labels, train_preds),
            'precision': precision_score(train_labels, train_preds, average='weighted'),
            'recall': recall_score(train_labels, train_preds, average='weighted'),
            'f1': f1_score(train_labels, train_preds, average='weighted')
        }
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Validation'):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                preds = output.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(target.cpu().numpy())
        
        val_metrics = {
            'loss': val_loss / len(val_loader),
            'accuracy': accuracy_score(val_labels, val_preds),
            'precision': precision_score(val_labels, val_preds, average='weighted'),
            'recall': recall_score(val_labels, val_preds, average='weighted'),
            'f1': f1_score(val_labels, val_preds, average='weighted')
        }
        
        # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # 更新历史记录
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['train_precision'].append(train_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['learning_rate'].append(current_lr)
        
        # 保存当前epoch的结果
        epoch_dir = os.path.join(save_dir, f'epoch_{epoch+1}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        # 保存混淆矩阵
        cm = confusion_matrix(val_labels, val_preds)
        plot_confusion_matrix(cm, class_names, os.path.join(epoch_dir, 'confusion_matrix.png'))
        
        # 保存指标
        metrics = {
            'train': train_metrics,
            'val': val_metrics,
            'learning_rate': current_lr
        }
        with open(os.path.join(epoch_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # 如果是最佳模型则保存
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': val_metrics
            }, os.path.join(save_dir, 'best_model.pth'))
            print("Saved new best model!")
        
        # 打印当前结果
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # 绘制并保存训练曲线
        plot_metrics(history, os.path.join(save_dir, 'training_curves.png'))
    
    # 保存完整训练历史
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)

if __name__ == '__main__':
    train()
