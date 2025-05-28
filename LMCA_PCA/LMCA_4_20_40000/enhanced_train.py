
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
import json
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from LMCA import LMCA
from training_utils import NetworkTrafficDataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

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
    
    # 绘制验证准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(epochs, [m['val_accuracy'] for m in metrics_history], label='Val Accuracy')
    plt.title('Validation Accuracy vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 绘制F1分数曲线
    plt.subplot(2, 2, 3)
    plt.plot(epochs, [m['val_f1'] for m in metrics_history], label='Val F1')
    plt.title('Validation F1 Score vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    
    # 绘制Precision和Recall
    plt.subplot(2, 2, 4)
    plt.plot(epochs, [m['val_precision'] for m in metrics_history], label='Val Precision')
    plt.plot(epochs, [m['val_recall'] for m in metrics_history], label='Val Recall')
    plt.title('Validation Precision & Recall vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_metrics.png'))
    plt.close()

class FocalLoss(nn.Module):
    """Focal Loss for dealing with class imbalance"""
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam



# 同时修改train_epoch_enhanced函数中的mixup部分
def train_epoch_enhanced(model, dataloader, criterion, optimizer, scheduler, device, mixup_alpha=0.2):
    """Enhanced training for one epoch with mixup and more metrics"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # Apply mixup
        mixed_data, target_a, target_b, lam = mixup_data(data, target, mixup_alpha)
        optimizer.zero_grad()
        output = model(mixed_data)
        loss = lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)
        
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        
        # For monitoring, compute predictions on original data
        with torch.no_grad():
            orig_output = model(data)
            preds = orig_output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(target.cpu().numpy())
        
        # 更新进度条
        pbar.set_postfix({'loss': f'{total_loss/(batch_idx+1):.4f}'})
    
    # 计算指标
    metrics = {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted'),
        'recall': recall_score(all_labels, all_preds, average='weighted'),
        'f1': f1_score(all_labels, all_preds, average='weighted')
    }
    
    return metrics

def validate_enhanced(model, dataloader, criterion, device):
    """Enhanced validation with more detailed metrics"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc='Validation'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            probs = F.softmax(output, dim=1)
            preds = output.argmax(dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    # 计算详细指标
    metrics = {
        'loss': total_loss / len(dataloader),
        'precision': precision_score(all_labels, all_preds, average='weighted'),
        'recall': recall_score(all_labels, all_preds, average='weighted'),
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='weighted')
    }
    
    # 分析困难样本
    probs_array = np.array(all_probs)
    max_probs = probs_array.max(axis=1)
    difficult_samples = (max_probs < 0.9).sum()
    metrics['difficult_samples'] = difficult_samples
    
    return metrics

def train_with_fold(fold_idx, train_loader, val_loader, config, save_dir):
    """Train one fold"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LMCA(num_classes=config['num_classes']).to(device)
    
    # 使用Focal Loss
    criterion = FocalLoss(gamma=config['focal_loss_gamma'])
    
    # 优化器和调度器
    optimizer = optim.Adam(model.parameters(), lr=config['initial_lr'], 
                          weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], 
                                 eta_min=config['min_lr'])
    
    best_val_f1 = 0
    metrics_history = []
    
    for epoch in range(config['epochs']):
        print(f"\nFold {fold_idx}, Epoch {epoch+1}/{config['epochs']}")
        
        # Training phase with mixup
        train_metrics = train_epoch_enhanced(
            model, train_loader, criterion, optimizer, scheduler, device, 
            mixup_alpha=config['mixup_alpha']
        )
        
        # Validation phase
        val_metrics = validate_enhanced(model, val_loader, criterion, device)
        
        # Collect metrics
        metrics = {
            'epoch': epoch + 1,
            'fold': fold_idx,
            'learning_rate': optimizer.param_groups[0]['lr'],
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()}
        }
        metrics_history.append(metrics)
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save({
                'fold': fold_idx,
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': metrics
            }, f'{save_dir}/fold_{fold_idx}_best.pth')
            print(f"Saved new best model for fold {fold_idx}!")
        
        # Print metrics
        print(f"Train - Loss: {train_metrics['loss']:.4f}")
        if 'accuracy' in train_metrics:
            print(f"Train - Accuracy: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val - F1: {val_metrics['f1']:.4f}, Difficult samples: {val_metrics['difficult_samples']}")
        
        # Save and plot metrics
        pd.DataFrame(metrics_history).to_csv(f'{save_dir}/fold_{fold_idx}_history.csv', index=False)
        plot_metrics(metrics_history, save_dir)
    
    return best_val_f1, metrics_history

def main():
    # Configuration
    config = {
        'data_dir': 'processed_data',
        'num_classes': 11,
        'batch_size': 256,
        'epochs': 20,
        'initial_lr': 1e-5,
        'min_lr': 1e-7,
        'weight_decay': 5e-4,
        'focal_loss_gamma': 2,
        'mixup_alpha': 0.1,
        'n_folds': 5
    }
    
    # Create save directory
    save_dir = 'enhanced_training_0.1'
    os.makedirs(save_dir, exist_ok=True)
    
    # Save configuration
    with open(f'{save_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Load dataset
    print("Loading dataset...")
    dataset = NetworkTrafficDataset(config['data_dir'], 'train')
    
    # K-fold cross validation
    kf = KFold(n_splits=config['n_folds'], shuffle=True, random_state=42)
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)))):
        print(f"\nTraining Fold {fold_idx+1}/{config['n_folds']}")
        
        # Create data loaders for this fold
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(dataset, batch_size=config['batch_size'],
                                sampler=train_sampler, num_workers=4)
        val_loader = DataLoader(dataset, batch_size=config['batch_size'],
                              sampler=val_sampler, num_workers=4)
        
        # Train this fold
        best_f1, history = train_with_fold(fold_idx, train_loader, val_loader, 
                                         config, save_dir)
        fold_results.append(best_f1)
    
    # Print final results
    print("\nTraining completed!")
    print("F1 scores for each fold:", fold_results)
    print(f"Average F1 score: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    
    # Save final results
    with open(f'{save_dir}/final_results.json', 'w') as f:
        json.dump({
            'fold_f1_scores': fold_results,
            'mean_f1': float(np.mean(fold_results)),
            'std_f1': float(np.std(fold_results))
        }, f, indent=4)

if __name__ == '__main__':
    main()


