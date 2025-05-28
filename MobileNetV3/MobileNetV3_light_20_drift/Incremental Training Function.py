
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import json
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

from IncrementalTrafficMobileNetV3 import create_incremental_model
from train import TrafficDataset, plot_confusion_matrix, plot_metrics

class CombinedLoss(nn.Module):
    def __init__(self, num_classes, temperature=0.07):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.num_classes = num_classes
        
    def forward(self, outputs, features, labels):
        # 交叉熵损失
        ce_loss = self.ce(outputs, labels)
        
        # 对比损失 - 使用batch内样本对
        batch_size = features.size(0)
        
        # 归一化特征
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        similarity = torch.matmul(features, features.t()) / self.temperature
        
        # 创建标签矩阵 - 只在batch内比较
        mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
        mask = mask - torch.eye(batch_size, device=features.device)  # 移除自身对角线
        
        # 计算正样本的loss
        positive_pairs = torch.exp(similarity) * mask
        denominator = torch.exp(similarity).sum(dim=1, keepdim=True)
        
        # 避免log(0)
        eps = 1e-8
        contrastive_loss = -torch.log(positive_pairs.sum(dim=1) / denominator.squeeze(1) + eps).mean()
        
        return ce_loss + 0.1 * contrastive_loss

def compute_ewc_loss(model, fisher, old_params):
    """改进的EWC损失计算"""
    ewc_loss = 0
    for name, param in model.named_parameters():
        if 'attention' in name or 'enhancement' in name:
            continue
            
        if name in fisher and name in old_params:
            if param.shape == old_params[name].shape:
                # 使用L1损失而不是L2
                ewc_loss += torch.sum(fisher[name] * torch.abs(param - old_params[name]))
            elif 'classifier' in name:
                old_size = old_params[name].size(0)
                ewc_loss += torch.sum(fisher[name][:old_size] * 
                                    torch.abs(param[:old_size] - old_params[name]))
    return ewc_loss

def get_layer_wise_lr_params(model):
    """获取差异化学习率参数"""
    params = []
    # 最后一层最大学习率
    params.append({
        'params': model.classifier[-1].parameters(),
        'lr': 5e-3
    })
    
    # 特征处理模块使用中等学习率
    feature_params = []
    for m in [model.feature_attention, model.feature_enhancement, 
              model.contrastive_head]:
        feature_params.extend(m.parameters())
    params.append({
        'params': feature_params,
        'lr': 1e-3
    })
    
    # 其他层使用基础学习率
    other_params = []
    for name, param in model.named_parameters():
        if not any(param is p for group in params for p in group['params']):
            other_params.append(param)
    params.append({
        'params': other_params,
        'lr': 5e-4
    })
    
    return params

def train_incremental():
    # 配置参数
    data_dir = 'processed_data_incremental_last_5'
    save_dir = 'training_results_incremental_optimized'
    num_epochs = 150
    batch_size = 256
    weight_decay = 1e-4
    ewc_lambda = 0.05
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置设备和精度
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    scaler = GradScaler()  # 混合精度训练
    
    # 加载Fisher信息和旧模型
    fisher_dict = torch.load('fisher_matrix.pth')
    fisher_matrix = fisher_dict['fisher_matrix']
    params_old = fisher_dict['optimal_params']
    
    # 创建模型
    model = create_incremental_model(
        base_model_path='model/best_model.pth',
        device=device
    )
    model.set_fisher_params(fisher_matrix, params_old)
    
    # 数据加载
    dataset = TrafficDataset(data_dir, 'train')
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          num_workers=4, pin_memory=True)
    
    # 损失函数和优化器
    criterion = CombinedLoss(num_classes=13).to(device)
    optimizer = optim.AdamW(get_layer_wise_lr_params(model), 
                           weight_decay=weight_decay)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[5e-3, 1e-3, 5e-4],  # 对应三个参数组
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 预热阶段
        anneal_strategy='cos'
    )
    
    # 训练历史
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [], 
        'val_loss': [], 'val_acc': [], 'val_f1': [], 
        'learning_rate': [], 'ewc_loss': []
    }
    
    best_val_f1 = 0
    patience = 10
    no_improve = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # 渐进式解冻
        if epoch < 20:
            model.freeze_old_params()
        elif epoch < 50:
            model.partial_unfreeze(3)
        else:
            model.unfreeze_all_params()
        
        # 训练阶段
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            
            with autocast():  # 混合精度训练
                outputs, features = model(data, return_features=True)
                loss = criterion(outputs, features, target)  # 更新损失计算
                # ce_loss = criterion(outputs, features, target)
                # ewc_loss = compute_ewc_loss(model, fisher_matrix, params_old)
                # loss = ce_loss + ewc_lambda * ewc_loss
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(target.cpu().numpy())
            
            # 打印批次级别的损失
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
        
        # 计算训练指标
        # 计算训练指标
        train_metrics = {
            'loss': train_loss / len(train_loader),
            'accuracy': accuracy_score(train_labels, train_preds),
            'f1': f1_score(train_labels, train_preds, average='weighted'),
            'precision': precision_score(train_labels, train_preds, average='weighted'),
            'recall': recall_score(train_labels, train_preds, average='weighted')
        }

        # 在history中添加
        
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader):
                data, target = data.to(device), target.to(device)
                
                outputs, features = model(data, return_features=True)
                loss = criterion(outputs, features, target)
                
                val_loss += loss.item()
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(target.cpu().numpy())
        
        # 计算验证指标
        val_metrics = {
            'loss': val_loss / len(val_loader),
            'accuracy': accuracy_score(val_labels, val_preds),
            'f1': f1_score(val_labels, val_preds, average='weighted')
        }
        
        # 更新历史记录
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        #history['train_precision'].append(train_metrics['precision'])
        #history['train_recall'].append(train_metrics['recall'])
        
        # 保存结果
        epoch_dir = os.path.join(save_dir, f'epoch_{epoch+1}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        # 混淆矩阵
        cm = confusion_matrix(val_labels, val_preds)
        plot_confusion_matrix(cm, range(13), 
                            os.path.join(epoch_dir, 'confusion_matrix.png'))
        
        # 保存指标
        metrics = {
            'train': train_metrics,
            'val': val_metrics,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        with open(os.path.join(epoch_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # 模型保存和早停
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            no_improve = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': val_metrics,
                'history': history
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"Saved new best model! F1: {best_val_f1:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered!")
                break
        
        # 打印当前结果
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}")
        
        # 绘制训练曲线
        # plot_metrics(history, os.path.join(save_dir, 'training_curves.png'))
    
    # 保存完整训练历史
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)

if __name__ == '__main__':
    train_incremental()
