import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import torch.nn.functional as F

from dense_res_opt_mobilenetv3 import TrafficMobileNetV3

def create_directories(base_dir='outputs'):
    """创建必要的目录"""
    dirs = {
        'checkpoint_dir': os.path.join(base_dir, 'checkpoints'),
        'confusion_matrix_dir': os.path.join(base_dir, 'confusion_matrices'),
        'reports_dir': os.path.join(base_dir, 'reports'),
        'logs_dir': os.path.join(base_dir, 'logs')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

class TrafficDataset(Dataset):
    """数据集加载器"""
    def __init__(self, data_dir, prefix, is_new_class=False):
        self.data_dir = data_dir
        self.prefix = prefix
        self.is_new_class = is_new_class
        
        # 加载数据
        self.features = []
        batch_idx = 0
        while True:
            batch_path = f"{data_dir}/{prefix}_features_batch_{batch_idx}.npy"
            if not os.path.exists(batch_path):
                break
            # 使用copy()确保数组的stride是正的
            self.features.append(np.load(batch_path).copy())
            batch_idx += 1
        
        self.features = np.concatenate(self.features, axis=0)
        self.labels = np.load(f"{data_dir}/{prefix}_labels.npy").copy()
        
        # 对新类别的标签进行偏移
        if is_new_class:
            self.labels = self.labels + 11
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx]).unsqueeze(0)
        label = torch.LongTensor([self.labels[idx]])[0]
        return feature, label, self.is_new_class

class FocalLoss(nn.Module):
    """Focal Loss处理类别不平衡"""
    def __init__(self, gamma=2, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if alpha is not None:
            self.alpha = torch.tensor(alpha)

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets].to(inputs.device)
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean()

def get_class_weights(dataset):
    """计算类别权重"""
    labels = [item[1].item() for item in dataset]
    class_counts = np.bincount(labels)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = 1.0 / class_counts
    
    # 提高BENIGN类的权重
    class_weights[0] *= 3.0
    # 稍微降低新类别的权重，避免过度拟合
    if len(class_weights) > 11:
        class_weights[11:] *= 3.0
    
    class_weights = class_weights / class_weights.sum()
    return torch.FloatTensor(class_weights)

def create_sampler(dataset, train_indices):
    """创建加权采样器"""
    labels = []
    for idx in train_indices:
        _, label, _ = dataset[idx]
        labels.append(label.item())
    
    class_counts = np.bincount(labels)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    weights = 1.0 / class_counts
    
    # 调整BENIGN类的采样权重
    weights[0] *= 1.5
    
    sample_weights = [weights[label] for label in labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_indices),
        replacement=True
    )

def calculate_fisher_loss(model, fisher_matrix, optimal_params):
    """计算Fisher正则化损失"""
    fisher_loss = 0
    for name, param in model.named_parameters():
        if ('classifier' in name or 
            name not in fisher_matrix or 
            name not in optimal_params or 
            param.shape != fisher_matrix[name].shape or 
            param.shape != optimal_params[name].shape):
            continue
        
        fisher_loss += (fisher_matrix[name] * (param - optimal_params[name]).pow(2)).sum()
    
    return fisher_loss

def elastic_loss(model, alpha=0.01):
    """弹性网络正则化"""
    l1_loss = 0
    l2_loss = 0
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            l1_loss += torch.abs(param).sum()
            l2_loss += (param ** 2).sum()
    return alpha * (0.5 * l2_loss + 0.5 * l1_loss)

def incremental_train():
    # 创建保存目录
    dirs = create_directories(base_dir='outputs')
    
    # 配置参数
    num_epochs = 30
    batch_size = 256
    base_lr = 1e-5
    classifier_lr = 1e-4
    weight_decay = 1e-5
    lambda_fisher = 5000.0
    lambda_distill = 2.0
    temperature = 2.0
    focal_gamma = 2.0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    old_dataset = TrafficDataset('processed_data_org', 'train', is_new_class=False)
    new_dataset = TrafficDataset('processed_data_incremental_last_5_3', 'train', is_new_class=True)
    
    print(f"Old dataset size: {len(old_dataset)}")
    print(f"New dataset size: {len(new_dataset)}")
    
    # 合并数据集
    combined_dataset = torch.utils.data.ConcatDataset([old_dataset, new_dataset])
    
    # 划分训练集和验证集
    indices = list(range(len(combined_dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_size = int(0.95 * len(indices))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # 创建数据加载器
    train_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        sampler=create_sampler(combined_dataset, train_indices),
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices),
        num_workers=4,
        pin_memory=True
    )
    
    # 初始化模型
    model = TrafficMobileNetV3(num_classes=12).to(device)
    
    # 加载预训练权重
    checkpoint_path = 'model/best_model.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        old_state_dict = checkpoint['model_state_dict']
        new_state_dict = model.state_dict()
        for name, param in old_state_dict.items():
            if 'classifier' not in name and name in new_state_dict:
                new_state_dict[name].copy_(param)
        print("Loaded pretrained weights successfully")
    
    # 设置优化器
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'classifier' not in n], 
         'lr': base_lr},
        {'params': [p for n, p in model.named_parameters() if 'classifier' in n], 
         'lr': classifier_lr}
    ]
    
    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2
    )
    
    # 设置损失函数
    class_weights = get_class_weights(combined_dataset).to(device)
    criterion = FocalLoss(gamma=focal_gamma, alpha=class_weights)
    
    # 加载Fisher矩阵
    fisher_path = 'fisher_matrix.pth'
    if os.path.exists(fisher_path):
        fisher_data = torch.load(fisher_path)
        fisher_matrix = fisher_data['fisher_matrix']
        optimal_params = fisher_data['optimal_params']
        print("Loaded Fisher matrix successfully")
    else:
        fisher_matrix = {}
        optimal_params = {}
    
    best_val_f1 = 0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # 训练阶段
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for data, target, is_new in tqdm(train_loader, desc='Training'):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # 前向传播
            output = model(data)
            
            # 分类损失
            loss = criterion(output, target)
            
            # Fisher正则化损失
            fisher_loss = calculate_fisher_loss(model, fisher_matrix, optimal_params)
            
            # 知识蒸馏损失
            if not is_new.all() and len(optimal_params) > 0:
                old_logits = output[:, :11]
                with torch.no_grad():
                    teacher_logits = F.softmax(old_logits / temperature, dim=1)
                student_logits = F.log_softmax(old_logits / temperature, dim=1)
                distill_loss = -(teacher_logits * student_logits).sum(dim=1).mean()
                distill_loss = distill_loss * (temperature ** 2)
            else:
                distill_loss = torch.tensor(0.0).to(device)
            
            # 弹性网络正则化
            elastic_regularization = elastic_loss(model)
            
            # 总损失
            total_loss = (loss + 
                         lambda_fisher * fisher_loss + 
                         lambda_distill * distill_loss + 
                         elastic_regularization)
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            preds = output.argmax(dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(target.cpu().numpy())
        
        scheduler.step()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for data, target, _ in tqdm(val_loader, desc='Validation'):
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                loss = criterion(output, target)
                val_loss += loss.item()
                
                preds = output.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(target.cpu().numpy())
        
        # 计算指标
        train_f1 = f1_score(train_labels, train_preds, average='weighted')
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val F1: {val_f1:.4f}")
        
        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = os.path.join(dirs['checkpoint_dir'], 'best_incremental_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_val_f1,
            }, save_path)
            
            # 保存混淆矩阵
            cm = confusion_matrix(val_labels, val_preds)
            plt.figure(figsize=(15, 12))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - Epoch {epoch+1}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(os.path.join(
                dirs['confusion_matrix_dir'], 
                f'confusion_matrix_epoch_{epoch+1}.png'
            ))
            plt.close()
            
            # 保存分类报告
            report = classification_report(val_labels, val_preds, digits=4)
            report_path = os.path.join(
                dirs['reports_dir'], 
                f'classification_report_epoch_{epoch+1}.txt'
            )
            with open(report_path, 'w') as f:
                f.write(report)
            print("\nClassification Report:")
            print(report)

if __name__ == '__main__':
    try:
        incremental_train()
        print("\nTraining completed successfully!")
    except Exception as e:
        print


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# import numpy as np
# import os
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, classification_report, f1_score
# import torch.nn.functional as F

# from dense_res_opt_mobilenetv3 import TrafficMobileNetV3

# # 创建必要的目录
# def create_directories(base_dir='outputs'):
#     checkpoint_dir = os.path.join(base_dir, 'checkpoints')
#     confusion_matrix_dir = os.path.join(base_dir, 'confusion_matrices')
#     reports_dir = os.path.join(base_dir, 'reports')
#     logs_dir = os.path.join(base_dir, 'logs')
    
#     os.makedirs(checkpoint_dir, exist_ok=True)
#     os.makedirs(confusion_matrix_dir, exist_ok=True)
#     os.makedirs(reports_dir, exist_ok=True)
#     os.makedirs(logs_dir, exist_ok=True)
    
#     return {
#         'checkpoint_dir': checkpoint_dir,
#         'confusion_matrix_dir': confusion_matrix_dir,
#         'reports_dir': reports_dir,
#         'logs_dir': logs_dir
#     }

# class TrafficDataset(Dataset):
#     def __init__(self, data_dir, prefix, is_new_class=False):
#         self.data_dir = data_dir
#         self.prefix = prefix
#         self.is_new_class = is_new_class
        
#         # 加载数据
#         self.features = []
#         batch_idx = 0
#         while True:
#             batch_path = f"{data_dir}/{prefix}_features_batch_{batch_idx}.npy"
#             if not os.path.exists(batch_path):
#                 break
#             self.features.append(np.load(batch_path))
#             batch_idx += 1
        
#         self.features = np.concatenate(self.features, axis=0)
#         self.labels = np.load(f"{data_dir}/{prefix}_labels.npy")
        
#         # 对新类别的标签进行偏移
#         if is_new_class:
#             self.labels = self.labels + 11
    
#     def __len__(self):
#         return len(self.features)
    
#     def __getitem__(self, idx):
#         feature = torch.FloatTensor(self.features[idx]).unsqueeze(0)
#         label = torch.LongTensor([self.labels[idx]])[0]
#         return feature, label, self.is_new_class

# class FocalLoss(nn.Module):
#     """Focal Loss来处理类别不平衡"""
#     def __init__(self, gamma=2, alpha=None):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if alpha is not None:
#             self.alpha = torch.tensor(alpha)

#     def forward(self, inputs, targets):
#         ce_loss = F.cross_entropy(inputs, targets, reduction='none')
#         pt = torch.exp(-ce_loss)
#         focal_loss = (1 - pt) ** self.gamma * ce_loss
        
#         if self.alpha is not None:
#             alpha_t = self.alpha[targets].to(inputs.device)
#             focal_loss = alpha_t * focal_loss
            
#         return focal_loss.mean()

# class WebAttackFeatureExtractor(nn.Module):
#     """专门用于Web攻击特征提取的模块"""
#     def __init__(self, in_channels):
#         super(WebAttackFeatureExtractor, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, 32, 3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#         )
        
#     def forward(self, x):
#         return self.conv(x)

# def get_class_weights(dataset):
#     """计算类别权重"""
#     labels = [item[1].item() for item in dataset]
#     class_counts = np.bincount(labels)
#     class_counts = np.where(class_counts == 0, 1, class_counts)
#     class_weights = 1.0 / class_counts
    
#     # 提高BENIGN类的权重
#     class_weights[0] *= 3.0
#     # 稍微降低新类别的权重，避免过度拟合
#     if len(class_weights) > 11:
#         class_weights[11:] *= 0.8
    
#     class_weights = class_weights / class_weights.sum()
#     return torch.FloatTensor(class_weights)

# def elastic_loss(model, alpha=1.0):
#     """添加弹性正则化"""
#     l1_loss = 0
#     l2_loss = 0
#     for name, param in model.named_parameters():
#         if 'classifier' not in name:  # 只对特征提取器应用
#             l1_loss += torch.abs(param).sum()
#             l2_loss += (param ** 2).sum()
#     return alpha * (0.5 * l2_loss + 0.5 * l1_loss)

# def create_sampler(dataset, train_indices):
#     """创建加权采样器"""
#     # 获取训练集中的标签
#     labels = []
#     for idx in train_indices:
#         _, label, _ = dataset[idx]
#         labels.append(label.item())
    
#     # 计算类别权重
#     class_counts = np.bincount(labels)
#     class_counts = np.where(class_counts == 0, 1, class_counts)  # 防止除以零
#     weights = 1.0 / class_counts
#     sample_weights = [weights[label] for label in labels]
    
#     # 创建采样器
#     return WeightedRandomSampler(
#         weights=sample_weights,
#         num_samples=len(train_indices),
#         replacement=True
#     )

# def calculate_fisher_loss(model, fisher_matrix, optimal_params):
#     """计算Fisher正则化损失，处理维度不匹配的情况"""
#     fisher_loss = 0
#     for name, param in model.named_parameters():
#         # 跳过分类头和维度不匹配的参数
#         if ('classifier' in name or 
#             name not in fisher_matrix or 
#             name not in optimal_params or 
#             param.shape != fisher_matrix[name].shape or 
#             param.shape != optimal_params[name].shape):
#             continue
        
#         fisher_loss += (fisher_matrix[name] * (param - optimal_params[name]).pow(2)).sum()
    
#     return fisher_loss

# def incremental_train():
#     # 创建保存目录
#     dirs = create_directories(base_dir='outputs')
    
#     # 配置参数
#     num_epochs = 50
#     batch_size = 256
#     base_lr = 1e-5          # 从 5e-5 降低到 1e-5
#     classifier_lr = 1e-4    # 从 5e-4 降低到 1e-4
#     weight_decay = 1e-5
#     lambda_fisher = 3000.0  # 降低Fisher正则化权重
#     lambda_distill = 2.0   # 降低知识蒸馏权重
#     temperature = 2.0
#     focal_gamma = 2.0
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
#     # 加载数据
#     old_dataset = TrafficDataset('processed_data_org', 'train', is_new_class=False)
#     new_dataset = TrafficDataset('processed_data_incremental_last_5_3', 'train', is_new_class=True)
    
#     # 打印数据集信息
#     print(f"Old dataset size: {len(old_dataset)}")
#     print(f"New dataset size: {len(new_dataset)}")
    
#     # 合并数据集
#     combined_dataset = torch.utils.data.ConcatDataset([old_dataset, new_dataset])
    
#     # 生成训练集和验证集的索引
#     indices = list(range(len(combined_dataset)))
#     np.random.seed(42)
#     np.random.shuffle(indices)
    
#     train_size = int(0.95 * len(indices))
#     train_indices = indices[:train_size]
#     val_indices = indices[train_size:]
    
#     # 创建采样器
#     train_sampler = create_sampler(combined_dataset, train_indices)
    
#     # 创建数据加载器
#     train_loader = DataLoader(
#         combined_dataset,
#         batch_size=batch_size,
#         sampler=train_sampler,
#         num_workers=4,
#         pin_memory=True
#     )
    
#     val_loader = DataLoader(
#         combined_dataset,
#         batch_size=batch_size,
#         sampler=torch.utils.data.SubsetRandomSampler(val_indices),
#         num_workers=4,
#         pin_memory=True
#     )
    
#     # 初始化模型
#     model = TrafficMobileNetV3(num_classes=12).to(device)
    
#     # 加载预训练权重
#     checkpoint_path = 'model/best_model.pth'
#     if os.path.exists(checkpoint_path):
#         checkpoint = torch.load(checkpoint_path, map_location=device)
#         # 加载除分类头外的权重
#         old_state_dict = checkpoint['model_state_dict']
#         new_state_dict = model.state_dict()
#         for name, param in old_state_dict.items():
#             if 'classifier' not in name and name in new_state_dict:
#                 new_state_dict[name].copy_(param)
#     else:
#         print(f"Checkpoint file '{checkpoint_path}' not found. Proceeding without loading pretrained weights.")
    
#     # 添加Web攻击特征提取器，输入通道数设为1（与原始输入相同）
#     web_attack_extractor = WebAttackFeatureExtractor(in_channels=1).to(device)
    
#     # 修改分类器以适应新的特征维度
#     # 获取原始特征维度
#     with torch.no_grad():
#         dummy_input = torch.randn(1, 1, 5, 5).to(device)  # 示例输入
#         original_features = model.get_features(dummy_input)
#         web_features = web_attack_extractor(dummy_input)
#         feature_dim = original_features.shape[1] + web_features.mean((2, 3)).shape[1]
    
#     # 重新定义分类器
#     model.classifier = nn.Sequential(
#         nn.Linear(feature_dim, 128),
#         nn.BatchNorm1d(128),
#         nn.ReLU(inplace=True),
#         nn.Dropout(0.2),
#         nn.Linear(128, 64),
#         nn.BatchNorm1d(64),
#         nn.ReLU(inplace=True),
#         nn.Dropout(0.2),
#         nn.Linear(64, 12)
#     ).to(device)
    
#     # 设置不同的参数组使用不同的学习率
#     param_groups = [
#         {'params': [p for n, p in model.named_parameters() if 'classifier' not in n], 'lr': base_lr},
#         {'params': model.classifier.parameters(), 'lr': classifier_lr},
#         {'params': web_attack_extractor.parameters(), 'lr': classifier_lr}
#     ]
    
#     optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
#     scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
#     # 计算类别权重用于Focal Loss
#     class_weights = get_class_weights(combined_dataset).to(device)
#     criterion = FocalLoss(gamma=focal_gamma, alpha=class_weights)
    
#     # 加载Fisher矩阵
#     fisher_path = 'fisher_matrix.pth'
#     if os.path.exists(fisher_path):
#         fisher_data = torch.load(fisher_path)
#         fisher_matrix = fisher_data['fisher_matrix']
#         optimal_params = fisher_data['optimal_params']
#     else:
#         print(f"Fisher matrix file '{fisher_path}' not found. Proceeding without Fisher regularization.")
#         fisher_matrix = {}
#         optimal_params = {}
    
#     best_val_f1 = 0
#     for epoch in range(num_epochs):
#         print(f"\nEpoch {epoch+1}/{num_epochs}")
        
#         # 训练阶段
#         model.train()
#         web_attack_extractor.train()
#         train_loss = 0
#         train_preds = []
#         train_labels = []
        
#         for batch_idx, (data, target, is_new) in enumerate(tqdm(train_loader, desc='Training')):
#             data, target = data.to(device), target.to(device)
#             optimizer.zero_grad()
            
#             # 前向传播
#             features = model.get_features(data)
#             web_features = web_attack_extractor(data)
#             combined_features = torch.cat([features, web_features.mean((2, 3))], dim=1)
#             output = model.classifier(combined_features)
            
#             # 计算分类损失
#             loss = criterion(output, target)
            
#             # 计算Fisher损失
#             fisher_loss = calculate_fisher_loss(model, fisher_matrix, optimal_params) if fisher_matrix else 0.0
            
#             # 知识蒸馏损失
#             if not is_new.all() and len(optimal_params) > 0:
#                 old_logits = output[:, :11]
#                 with torch.no_grad():
#                     teacher_logits = F.softmax(old_logits / temperature, dim=1)
#                 student_logits = F.log_softmax(old_logits / temperature, dim=1)
#                 distill_loss = -(teacher_logits * student_logits).sum(dim=1).mean() * (temperature ** 2)
#             else:
#                 distill_loss = torch.tensor(0.0).to(device)
            
#             # Web攻击类别的对比损失
#             contrast_loss = torch.tensor(0.0).to(device)
#             web_attack_mask = (target >= 11) & (target <= 12)
#             if web_attack_mask.sum() > 1:  # 至少需要两个Web攻击样本
#                 web_features_selected = combined_features[web_attack_mask]
#                 web_labels_selected = target[web_attack_mask]
                
#                 # 计算特征相似度
#                 web_features_norm = F.normalize(web_features_selected, dim=1)
#                 similarity = torch.matmul(web_features_norm, web_features_norm.t())
                
#                 # 创建标签矩阵
#                 labels_matrix = web_labels_selected.unsqueeze(0) == web_labels_selected.unsqueeze(1)
                
#                 # 计算对比损失
#                 mask = torch.eye(labels_matrix.shape[0], device=device) * -1e9
#                 similarity = similarity + mask
                
#                 positive_pairs = labels_matrix.float() * similarity
#                 negative_pairs = (1 - labels_matrix.float()) * similarity
                
#                 contrast_loss = (-torch.logsumexp(positive_pairs, dim=1) + 
#                                torch.logsumexp(negative_pairs, dim=1)).mean()
            
#             # 总损失
#             elastic_regularization = elastic_loss(model, alpha=0.01)
#             total_loss = (loss + 
#                         lambda_fisher * fisher_loss + 
#                         lambda_distill * distill_loss + 
#                         0.1 * contrast_loss +
#                         elastic_regularization)
            
#             total_loss.backward()
#             optimizer.step()
            
#             train_loss += total_loss.item()
#             preds = output.argmax(dim=1).cpu().numpy()
#             train_preds.extend(preds)
#             train_labels.extend(target.cpu().numpy())
        
#         # 更新学习率
#         scheduler.step()
        
#         # 验证阶段
#         model.eval()
#         web_attack_extractor.eval()
#         val_loss = 0
#         val_preds = []
#         val_labels = []
        
#         with torch.no_grad():
#             for data, target, _ in tqdm(val_loader, desc='Validation'):
#                 data, target = data.to(device), target.to(device)
#                 features = model.get_features(data)
#                 web_features = web_attack_extractor(data)
#                 combined_features = torch.cat([features, web_features.mean((2, 3))], dim=1)
#                 output = model.classifier(combined_features)
                
#                 loss = criterion(output, target)
#                 val_loss += loss.item()
                
#                 preds = output.argmax(dim=1).cpu().numpy()
#                 val_preds.extend(preds)
#                 val_labels.extend(target.cpu().numpy())
        
#         # 计算并打印指标
#         train_f1 = f1_score(train_labels, train_preds, average='weighted')
#         val_f1 = f1_score(val_labels, val_preds, average='weighted')
        
#         print(f"\nTrain Loss: {train_loss/len(train_loader):.4f}, Train F1: {train_f1:.4f}")
#         print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val F1: {val_f1:.4f}")
        
#         # 保存最佳模型
#         if val_f1 > best_val_f1:
#             best_val_f1 = val_f1
#             checkpoint_save_path = os.path.join(dirs['checkpoint_dir'], 'best_incremental_model.pth')
#             torch.save({
#                 'epoch': epoch + 1,
#                 'model_state_dict': model.state_dict(),
#                 'web_attack_extractor_state_dict': web_attack_extractor.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'scheduler_state_dict': scheduler.state_dict(),
#                 'best_f1': best_val_f1,
#             }, checkpoint_save_path)
#             print(f"Saved best model to {checkpoint_save_path}")
            
#             # 保存混淆矩阵
#             cm = confusion_matrix(val_labels, val_preds)
#             plt.figure(figsize=(15, 12))
#             sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#             plt.title(f'Confusion Matrix - Epoch {epoch+1}')
#             plt.ylabel('True Label')
#             plt.xlabel('Predicted Label')
#             plt.tight_layout()
#             cm_save_path = os.path.join(dirs['confusion_matrix_dir'], f'confusion_matrix_epoch_{epoch+1}.png')
#             plt.savefig(cm_save_path)
#             plt.close()
#             print(f"Saved confusion matrix to {cm_save_path}")
            
#             # 打印Web攻击类别的详细指标
#             web_attack_indices = [11, 12]  # Web Attack类别的索引
#             web_attack_labels = [l for l, p in zip(val_labels, val_preds) if l in web_attack_indices]
#             web_attack_preds = [p for l, p in zip(val_labels, val_preds) if l in web_attack_indices]
#             if web_attack_labels and web_attack_preds:
#                 web_attack_report = classification_report(
#                     web_attack_labels,
#                     web_attack_preds,
#                     digits=4
#                 )
#                 report_save_path = os.path.join(dirs['reports_dir'], f'web_attack_report_epoch_{epoch+1}.txt')
#                 with open(report_save_path, 'w') as f:
#                     f.write(web_attack_report)
#                 print(f"Saved Web Attack Classification Report to {report_save_path}")
#                 print("\nWeb Attack Classification Report:")
#                 print(web_attack_report)
#             else:
#                 print("No Web Attack samples in validation set for this epoch.")

# if __name__ == '__main__':
#     incremental_train()