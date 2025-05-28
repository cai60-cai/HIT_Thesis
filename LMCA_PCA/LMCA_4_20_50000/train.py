
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from LMCA import LMCA  # 导入模型

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

def plot_metrics(metrics_history, save_dir='models'):
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

def main():
    # 配置参数
    data_dir = 'processed_data'
    num_classes = 11
    batch_size = 256
    num_epochs = 20
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建保存目录
    os.makedirs('models', exist_ok=True)
    
    # 加载数据
    print("Loading dataset...")
    dataset = NetworkTrafficDataset(data_dir, 'train')
    
    # 划分训练集和验证集 (95% : 5%)
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=4)
    
    # 初始化模型
    model = LMCA(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # 学习率调度
    lr_schedule = [
        {'epochs': 5, 'lr': 1e-4},
        {'epochs': 5, 'lr': 1e-5},
        {'epochs': 10, 'lr': 1e-6}
    ]
    
    # 训练循环
    current_epoch = 0
    best_val_acc = 0
    metrics_history = []
    
    for schedule in lr_schedule:
        lr = schedule['lr']
        epochs = schedule['epochs']
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        
        for epoch in range(epochs):
            current_epoch += 1
            print(f"\nEpoch {current_epoch}/{num_epochs}")
            print(f"Learning rate: {lr}")
            
            # 创建当前epoch的保存目录
            epoch_dir = f'models_avg/epoch_{current_epoch}'
            os.makedirs(epoch_dir, exist_ok=True)
            
            # 训练和验证
            train_loss, train_prec, train_recall, train_acc, train_f1 = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            
            val_loss, val_prec, val_recall, val_acc, val_f1 = validate(
                model, val_loader, criterion, device
            )
            
            # 收集指标
            metrics = {
                'epoch': current_epoch,
                'learning_rate': lr,
                'train_loss': train_loss,
                'train_precision': train_prec,
                'train_recall': train_recall,
                'train_accuracy': train_acc,
                'train_f1': train_f1,
                'val_loss': val_loss,
                'val_precision': val_prec,
                'val_recall': val_recall,
                'val_accuracy': val_acc,
                'val_f1': val_f1
            }
            metrics_history.append(metrics)
            
            # 保存当前epoch的指标
            with open(f'{epoch_dir}/metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)
            
            # 保存当前epoch的模型
            torch.save({
                'epoch': current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'metrics': metrics
            }, f'{epoch_dir}/model.pth')
            
            # 如果是最佳模型则保存
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': current_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'metrics': metrics
                }, 'models/best_model.pth')
                print("Saved new best model!")
            
            # 打印当前epoch的指标
            print(f"\nTrain - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
            print(f"Train - Precision: {train_prec:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
            print(f"Val - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
            print(f"Val - Precision: {val_prec:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
            
            # 绘制并保存训练曲线
            plot_metrics(metrics_history)
    
    # 保存完整的训练历史
    import pandas as pd
    pd.DataFrame(metrics_history).to_csv('models/training_history.csv', index=False)
    
    print("\nTraining completed!")
    print(f"Best validation F1: {best_val_acc:.4f}")
    print("All models and metrics have been saved in the 'models' directory")

if __name__ == '__main__':
    main()
