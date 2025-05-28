import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import numpy as np
import os
import json
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from train import plot_metrics, plot_confusion_matrix, TrafficDataset
import torch.nn.functional as F
from enhanced_model import IncrementalTrafficMobileNetV3

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha)
        self.register_buffer('alpha', alpha)
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1-pt)**self.gamma * ce_loss
        else:
            focal_loss = (1-pt)**self.gamma * ce_loss
        
        return focal_loss.mean()

class IncrementalTrafficDataset(Dataset):
    """增量数据集加载器"""
    def __init__(self, data_dir, prefix, is_new_class=False, label_offset=0):
        super().__init__()
        self.data_dir = data_dir
        self.prefix = prefix
        self.is_new_class = is_new_class
        
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

def train_incremental_13_with_port_scan_weights():
    # 配置参数
    old_data_dir = 'processed_data_org'  # 原始数据
    first_increment_dir = 'processed_data_incremental_last_5_3'  # 第一次增量数据
    newest_data_dir = 'processed_data_incremental_last_5_4'  # 最新类别数据
    base_save_dir = 'training_results_incremental_13_port_scan_weights'
    old_model_path = 'training_results_incremental_yuohua_webmokuai/best_model.pth'  # 12类模型
    
    # 配置训练参数
    num_epochs = 30
    batch_size = 256
    learning_rate = 1e-4
    weight_decay = 1e-5
    temperature = 2.0
    alpha = 0.5
    val_split_ratio = 0.05  # 5% validation split
    
    # 设置端口扫描权重范围
    port_scan_weights = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    
    # 创建基础保存目录
    os.makedirs(base_save_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载所有数据集
    old_train_dataset = TrafficDataset(old_data_dir, 'train')  # 原始类别
    
    # 第一次增量的数据(11-12类)
    first_increment_train = IncrementalTrafficDataset(first_increment_dir, 'train', is_new_class=True, label_offset=11)
    
    # 最新类别数据(13类)
    new_train_dataset = IncrementalTrafficDataset(newest_data_dir, 'train', is_new_class=True, label_offset=12)
    
    # 对每个权重进行训练
    for port_scan_weight in port_scan_weights:
        # 创建当前权重的保存目录
        current_save_dir = os.path.join(base_save_dir, f'port_scan_weight_{port_scan_weight}')
        os.makedirs(current_save_dir, exist_ok=True)
        
        # 合并数据集
        combined_train_dataset = ConcatDataset([old_train_dataset, first_increment_train, new_train_dataset])
        
        # 手动划分训练和验证集
        val_size = int(len(combined_train_dataset) * val_split_ratio)
        train_size = len(combined_train_dataset) - val_size
        train_dataset, val_dataset = random_split(combined_train_dataset, [train_size, val_size])
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # 初始化模型
        model = IncrementalTrafficMobileNetV3(old_model_path, old_num_classes=12, new_num_classes=13).to(device)
        
        # 设置类别权重
        class_weights = torch.ones(13)
        class_weights[9] = port_scan_weight  # PortScan的权重
        class_weights[11] = 1.0  # 第一次增量类别的权重
        class_weights[12] = 1.0  # 最新类别的权重
        
        # 打印当前权重配置
        print(f"\n开始训练 - PortScan权重: {port_scan_weight}")
        print(f"类别权重: {class_weights}")
        
        # 损失函数
        criterion = FocalLoss(gamma=2, alpha=class_weights).to(device)
        distillation_criterion = nn.KLDivLoss(reduction='batchmean')
        
        # 优化器
        optimizer = optim.Adam([
            {'params': model.conv2.parameters()},
            {'params': model.classifier.parameters()}
        ], lr=learning_rate, weight_decay=weight_decay)
        
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
        
        def calculate_loss(outputs, labels, old_outputs=None):
            """计算总损失"""
            focal_loss = criterion(outputs, labels)
            if old_outputs is not None:
                old_probs = F.softmax(old_outputs / temperature, dim=1)
                new_log_probs = F.log_softmax(outputs[:, :12] / temperature, dim=1)
                distill_loss = distillation_criterion(new_log_probs, old_probs)
                return focal_loss + alpha * (temperature ** 2) * distill_loss
            return focal_loss
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # 训练阶段
            model.train()
            train_loss = 0
            train_preds = []
            train_labels = []
            
            for data, target in tqdm(train_loader, desc='Training'):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                
                # 获取新旧模型输出
                old_output, new_output = model.get_old_new_outputs(data)
                
                # 计算损失
                loss = calculate_loss(new_output, target, old_output)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                preds = new_output.argmax(dim=1).cpu().numpy()
                train_preds.extend(preds)
                train_labels.extend(target.cpu().numpy())
            
            train_metrics = {
                'loss': train_loss / len(train_loader),
                'acc': accuracy_score(train_labels, train_preds),
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
                    old_output, new_output = model.get_old_new_outputs(data)
                    loss = calculate_loss(new_output, target, old_output)
                    
                    val_loss += loss.item()
                    preds = new_output.argmax(dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_labels.extend(target.cpu().numpy())
            
            val_metrics = {
                'loss': val_loss / len(val_loader),
                'acc': accuracy_score(val_labels, val_preds),
                'precision': precision_score(val_labels, val_preds, average='weighted'),
                'recall': recall_score(val_labels, val_preds, average='weighted'),
                'f1': f1_score(val_labels, val_preds, average='weighted')
            }
            
            # 更新学习率
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            
            # 更新历史记录
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['acc'])
            history['train_f1'].append(train_metrics['f1'])
            history['train_precision'].append(train_metrics['precision'])
            history['train_recall'].append(train_metrics['recall'])
            
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['acc'])
            history['val_f1'].append(val_metrics['f1'])
            history['val_precision'].append(val_metrics['precision'])
            history['val_recall'].append(val_metrics['recall'])
            
            history['learning_rate'].append(current_lr)
            
            # 保存当前epoch的结果
            epoch_dir = os.path.join(current_save_dir, f'epoch_{epoch+1}')
            os.makedirs(epoch_dir, exist_ok=True)
            
            # 保存混淆矩阵
            cm = confusion_matrix(val_labels, val_preds)
            class_names = [str(i) for i in range(13)]  # 13个类别
            plot_confusion_matrix(cm, class_names, os.path.join(epoch_dir, 'confusion_matrix.png'))
            
            # 保存指标
            metrics = {
                'train': train_metrics,
                'val': val_metrics,
                'learning_rate': current_lr,
                'port_scan_weight': port_scan_weight
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
                    'metrics': val_metrics,
                    'old_num_classes': model.old_num_classes,
                    'new_num_classes': model.new_num_classes,
                    'port_scan_weight': port_scan_weight
                }, os.path.join(current_save_dir, 'best_model.pth'))
                print("Saved new best model!")
            
            # 打印当前结果
            print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}, F1: {val_metrics['f1']:.4f}")
            
            # 绘制并保存训练曲线
            plot_metrics(history, os.path.join(current_save_dir, 'training_curves.png'))
        
        # 保存完整训练历史
        with open(os.path.join(current_save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=4)
        
        # 释放CUDA内存
        torch.cuda.empty_cache()

if __name__ == '__main__':
    train_incremental_13_with_port_scan_weights()