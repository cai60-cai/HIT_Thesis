# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader, Subset
# import numpy as np
# import os
# import json
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
# from train import plot_confusion_matrix, plot_metrics
# from dense_res_opt_mobilenetv3 import TrafficMobileNetV3
# import pandas as pd
# from typing import List, Dict, Any

# class IncrementalTrafficDataset(Dataset):
#     """增量学习数据集加载器"""
#     def __init__(self, data_dir: str, prefix: str):
#         self.data_dir = data_dir
#         self.prefix = prefix
        
#         # 加载所有批次特征
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
        
#         # 加载标签映射
#         label_mapping_df = pd.read_csv(f"{data_dir}/label_mapping.csv", index_col=0)
#         self.label_mapping = {label: idx for idx, label in enumerate(label_mapping_df.index)}
        
#     def __len__(self):
#         return len(self.features)
    
#     def __getitem__(self, idx):
#         feature = torch.FloatTensor(self.features[idx]).unsqueeze(0)
#         label = torch.LongTensor([self.labels[idx]])[0]
#         return feature, label

# class WeightedDistillationLoss(nn.Module):
#     """加权知识蒸馏损失"""
#     def __init__(self, num_classes, drift_classes, label_mapping, temperature=2.0):
#         super().__init__()
#         self.temperature = temperature
#         self.kl_div = nn.KLDivLoss(reduction='none')
#         self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
#         # 设置类别权重
#         self.class_weights = torch.ones(num_classes)
#         for class_name, idx in label_mapping.items():
#             if class_name not in drift_classes:
#                 self.class_weights[idx] = 2.0  # 非漂移类别权重更高
    
#     def forward(self, student_logits, teacher_logits, labels, alpha=0.5):
#         device = student_logits.device
#         self.class_weights = self.class_weights.to(device)
        
#         # 计算蒸馏损失
#         soft_targets = torch.nn.functional.softmax(teacher_logits / self.temperature, dim=1)
#         soft_prob = torch.nn.functional.log_softmax(student_logits / self.temperature, dim=1)
#         distillation_loss = self.kl_div(soft_prob, soft_targets).mean(dim=1)
        
#         # 计算交叉熵损失
#         ce_loss = self.ce_loss(student_logits, labels)
#         weights = self.class_weights[labels]
        
#         # 组合损失
#         total_loss = alpha * distillation_loss * (self.temperature ** 2) + \
#                     (1 - alpha) * ce_loss * weights
        
#         return total_loss.mean()

# def get_drift_classes():
#     """获取漂移类别列表"""
#     dos_ddos_classes = ['DoS slowloris', 'DDoS', 'DoS Hulk']
#     noise_classes = ['BENIGN', 'PortScan', 'SSH-Patator', 'Infiltration']
#     return dos_ddos_classes + noise_classes

# def create_balanced_indices(dataset: IncrementalTrafficDataset):
#     """创建平衡的训练索引"""
#     all_indices = []
#     drift_classes = get_drift_classes()
    
#     # 获取所有类别
#     all_classes = list(dataset.label_mapping.keys())
#     non_drift_classes = [c for c in all_classes if c not in drift_classes]
    
#     # 对于漂移类别，使用全部数据
#     for class_name in drift_classes:
#         class_idx = dataset.label_mapping[class_name]
#         class_indices = np.where(dataset.labels == class_idx)[0]
#         all_indices.extend(class_indices)
    
#     # 对于非漂移类别，采样部分数据
#     sample_ratio = 0.3
#     for class_name in non_drift_classes:
#         class_idx = dataset.label_mapping[class_name]
#         class_indices = np.where(dataset.labels == class_idx)[0]
#         sampled_indices = np.random.choice(
#             class_indices, 
#             size=int(len(class_indices) * sample_ratio), 
#             replace=False
#         )
#         all_indices.extend(sampled_indices)
    
#     return np.array(all_indices)

# def incremental_train(
#     original_model_path: str,
#     data_dir: str = 'processed_data_drift',
#     save_dir: str = 'incremental_training_results_yuanshibujiatezhengceng',
#     num_epochs: int = 30,
#     batch_size: int = 256,
#     learning_rate: float = 1e-4,
#     weight_decay: float = 1e-5,
#     distill_alpha: float = 0.5
# ):
#     # 创建保存目录
#     os.makedirs(save_dir, exist_ok=True)
    
#     # 设置设备
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     # 加载数据集
#     dataset = IncrementalTrafficDataset(data_dir, 'train')
    
#     # 获取漂移类别
#     drift_classes = get_drift_classes()
#     print(f"Drift classes: {drift_classes}")
    
#     # 创建平衡的数据索引
#     balanced_indices = create_balanced_indices(dataset)
#     np.random.shuffle(balanced_indices)
    
#     # 划分训练集和验证集 (90:10)
#     train_size = int(0.9 * len(balanced_indices))
#     train_indices = balanced_indices[:train_size]
#     val_indices = balanced_indices[train_size:]
    
#     # 创建数据加载器
#     train_dataset = Subset(dataset, train_indices)
#     val_dataset = Subset(dataset, val_indices)
    
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
#     # 加载模型
#     model = TrafficMobileNetV3(num_classes=len(dataset.label_mapping))
#     checkpoint = torch.load(original_model_path, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model = model.to(device)
    
#     # 创建教师模型
#     teacher_model = TrafficMobileNetV3(num_classes=len(dataset.label_mapping))
#     teacher_model.load_state_dict(checkpoint['model_state_dict'])
#     teacher_model = teacher_model.to(device)
#     teacher_model.eval()
    
#     # 创建损失函数和优化器
#     criterion = WeightedDistillationLoss(
#         num_classes=len(dataset.label_mapping),
#         drift_classes=drift_classes,
#         label_mapping=dataset.label_mapping
#     )
    
#     # 仅优化漂移类别的分类层参数
#     trainable_params = []
#     for name, param in model.named_parameters():
#         if 'classifier' in name:
#             param.requires_grad = True
#             trainable_params.append(param)
#         else:
#             param.requires_grad = False
    
#     optimizer = optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
#     # 训练历史
#     history = {
#         'train_loss': [], 'train_acc': [], 'train_f1': [], 
#         'train_precision': [], 'train_recall': [],
#         'val_loss': [], 'val_acc': [], 'val_f1': [], 
#         'val_precision': [], 'val_recall': [],
#         'learning_rate': []
#     }
    
#     best_val_f1 = 0
#     current_lr = optimizer.param_groups[0]['lr']
    
#     for epoch in range(num_epochs):
#         print(f"\nEpoch {epoch+1}/{num_epochs}")
#         print(f"Current learning rate: {current_lr}")
        
#         # 训练阶段
#         model.train()
#         train_loss = 0
#         train_preds = []
#         train_labels = []
        
#         for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc='Training')):
#             data, target = data.to(device), target.to(device)
#             optimizer.zero_grad()
            
#             student_output = model(data)
#             with torch.no_grad():
#                 teacher_output = teacher_model(data)
            
#             loss = criterion(student_output, teacher_output, target, alpha=distill_alpha)
#             loss.backward()
#             optimizer.step()
            
#             train_loss += loss.item()
#             preds = student_output.argmax(dim=1).cpu().numpy()
#             train_preds.extend(preds)
#             train_labels.extend(target.cpu().numpy())
        
#         # 计算训练指标
#         train_metrics = {
#             'loss': train_loss / len(train_loader),
#             'accuracy': accuracy_score(train_labels, train_preds),
#             'precision': precision_score(train_labels, train_preds, average='weighted', zero_division=1),
#             'recall': recall_score(train_labels, train_preds, average='weighted', zero_division=1),
#             'f1': f1_score(train_labels, train_preds, average='weighted', zero_division=1)
#         }
        
#         # 验证阶段
#         model.eval()
#         val_loss = 0
#         val_preds = []
#         val_labels = []
        
#         with torch.no_grad():
#             for data, target in tqdm(val_loader, desc='Validation'):
#                 data, target = data.to(device), target.to(device)
#                 student_output = model(data)
#                 teacher_output = teacher_model(data)
                
#                 loss = criterion(student_output, teacher_output, target, alpha=distill_alpha)
#                 val_loss += loss.item()
                
#                 preds = student_output.argmax(dim=1).cpu().numpy()
#                 val_preds.extend(preds)
#                 val_labels.extend(target.cpu().numpy())
        
#         # 计算验证指标
#         val_metrics = {
#             'loss': val_loss / len(val_loader),
#             'accuracy': accuracy_score(val_labels, val_preds),
#             'precision': precision_score(val_labels, val_preds, average='weighted', zero_division=1),
#             'recall': recall_score(val_labels, val_preds, average='weighted', zero_division=1),
#             'f1': f1_score(val_labels, val_preds, average='weighted', zero_division=1)
#         }
        
#         # 更新历史记录
#         history['train_loss'].append(train_metrics['loss'])
#         history['train_acc'].append(train_metrics['accuracy'])
#         history['train_f1'].append(train_metrics['f1'])
#         history['train_precision'].append(train_metrics['precision'])
#         history['train_recall'].append(train_metrics['recall'])
        
#         history['val_loss'].append(val_metrics['loss'])
#         history['val_acc'].append(val_metrics['accuracy'])
#         history['val_f1'].append(val_metrics['f1'])
#         history['val_precision'].append(val_metrics['precision'])
#         history['val_recall'].append(val_metrics['recall'])
#         history['learning_rate'].append(current_lr)
        
#         # 更新学习率
#         scheduler.step()
#         current_lr = optimizer.param_groups[0]['lr']
        
#         # 保存当前epoch的结果
#         epoch_dir = os.path.join(save_dir, f'epoch_{epoch+1}')
#         os.makedirs(epoch_dir, exist_ok=True)
        
#         # 保存混淆矩阵
#         cm = confusion_matrix(val_labels, val_preds)
#         plot_confusion_matrix(cm, list(dataset.label_mapping.keys()), 
#                             os.path.join(epoch_dir, 'confusion_matrix.png'))
        
#         # 保存指标
#         metrics = {
#             'train': train_metrics,
#             'val': val_metrics,
#             'learning_rate': current_lr
#         }
#         with open(os.path.join(epoch_dir, 'metrics.json'), 'w') as f:
#             json.dump(metrics, f, indent=4)
        
#         # 保存最佳模型
#         if val_metrics['f1'] > best_val_f1:
#             best_val_f1 = val_metrics['f1']
#             torch.save({
#                 'epoch': epoch + 1,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'scheduler_state_dict': scheduler.state_dict(),
#                 'metrics': val_metrics
#             }, os.path.join(save_dir, 'best_model.pth'))
#             print("Saved new best model!")
        
#         # 打印当前结果
#         print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
#         print(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
        
#         # 绘制训练曲线
#         plot_metrics(history, os.path.join(save_dir, 'training_curves.png'))
    
#     # 保存训练历史
#     with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
#         json.dump(history, f, indent=4)

# if __name__ == '__main__':
#     # 配置参数
#     config = {
#         'original_model_path': 'model/best_model.pth',  # 原始模型路径
#         'data_dir': 'processed_data_drift',  # 漂移数据目录
#         'save_dir': 'incremental_training_results_yuanshibujiatezhengceng',  # 结果保存目录
#         'num_epochs': 50,  # 训练轮数
#         'batch_size': 256,  # 批次大小
#         'learning_rate': 1e-4,  # 学习率
#         'weight_decay': 1e-5,  # 权重衰减
#         'distill_alpha': 0.5  # 知识蒸馏权重
#     }
    
#     print("Starting incremental learning with configurations:")
#     for key, value in config.items():
#         print(f"{key}: {value}")
    
#     # 运行增量训练
#     incremental_train(
#         original_model_path=config['original_model_path'],
#         data_dir=config['data_dir'],
#         save_dir=config['save_dir'],
#         num_epochs=config['num_epochs'],
#         batch_size=config['batch_size'],
#         learning_rate=config['learning_rate'],
#         weight_decay=config['weight_decay'],
#         distill_alpha=config['distill_alpha']
#     )



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from dense_res_opt_mobilenetv3 import TrafficMobileNetV3
import pandas as pd
from typing import List, Dict, Any
from torch.cuda.amp import autocast, GradScaler

class IncrementalTrafficDataset(Dataset):
    """增量学习数据集加载器"""
    def __init__(self, data_dir: str, prefix: str):
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
        
        # 加载标签映射
        label_mapping_df = pd.read_csv(f"{data_dir}/label_mapping.csv", index_col=0)
        self.label_mapping = {label: idx for idx, label in enumerate(label_mapping_df.index)}
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx]).unsqueeze(0)
        label = torch.LongTensor([self.labels[idx]])[0]
        return feature, label

class FeatureAttention(nn.Module):
    """特征注意力机制"""
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        weights = self.attention(x)
        return x * weights

class WeightedDistillationLoss(nn.Module):
    """加权知识蒸馏损失"""
    def __init__(self, num_classes, drift_classes, label_mapping, temperature=2.0):
        super().__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
        # 设置类别权重
        self.class_weights = torch.ones(num_classes)
        focus_classes = ['Heartbleed', 'Infiltration', 'PortScan', 'SSH-Patator']
        
        for class_name, idx in label_mapping.items():
            if class_name not in drift_classes:
                self.class_weights[idx] = 1.5  # 非漂移类别基础权重
            if class_name in focus_classes:
                self.class_weights[idx] = 3.0  # 重点关注类别权重
    
    def forward(self, student_logits, teacher_logits, labels, alpha=0.5):
        device = student_logits.device
        self.class_weights = self.class_weights.to(device)
        
        # 计算蒸馏损失
        soft_targets = torch.nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        soft_prob = torch.nn.functional.log_softmax(student_logits / self.temperature, dim=1)
        distillation_loss = self.kl_div(soft_prob, soft_targets).mean(dim=1)
        
        # 计算交叉熵损失
        ce_loss = self.ce_loss(student_logits, labels)
        weights = self.class_weights[labels]
        
        # 组合损失
        total_loss = alpha * distillation_loss * (self.temperature ** 2) + \
                    (1 - alpha) * ce_loss * weights
        
        return total_loss.mean()

def get_drift_classes():
    """获取漂移类别列表"""
    dos_ddos_classes = ['DoS slowloris', 'DDoS', 'DoS Hulk']
    noise_classes = ['BENIGN', 'PortScan', 'SSH-Patator', 'Infiltration']
    return dos_ddos_classes + noise_classes

def create_balanced_indices(dataset: IncrementalTrafficDataset):
    """创建平衡的训练索引，重点关注问题类别"""
    all_indices = []
    drift_classes = get_drift_classes()
    focus_classes = ['Heartbleed', 'Infiltration', 'PortScan', 'SSH-Patator']
    
    # 重点类别过采样
    for class_name in focus_classes:
        class_idx = dataset.label_mapping[class_name]
        class_indices = np.where(dataset.labels == class_idx)[0]
        if len(class_indices) > 0:
            sampled_indices = np.random.choice(
                class_indices,
                size=int(len(class_indices) * 2),
                replace=True
            )
            all_indices.extend(sampled_indices)
    
    # 其他漂移类别正常采样
    other_drift = [c for c in drift_classes if c not in focus_classes]
    for class_name in other_drift:
        class_idx = dataset.label_mapping[class_name]
        class_indices = np.where(dataset.labels == class_idx)[0]
        all_indices.extend(class_indices)
    
    # 非漂移类别少量采样
    all_classes = list(dataset.label_mapping.keys())
    non_drift_classes = [c for c in all_classes if c not in drift_classes]
    for class_name in non_drift_classes:
        class_idx = dataset.label_mapping[class_name]
        class_indices = np.where(dataset.labels == class_idx)[0]
        sampled_indices = np.random.choice(
            class_indices,
            size=int(len(class_indices) * 0.2),
            replace=False
        )
        all_indices.extend(sampled_indices)
    
    return np.array(all_indices)

def plot_metrics(history, save_path):
    """绘制训练指标曲线"""
    plt.figure(figsize=(20, 12))
    
    metrics = [
        ('loss', 'Loss'),
        ('acc', 'Accuracy'),
        ('f1', 'F1 Score'),
        ('precision', 'Precision'),
        ('recall', 'Recall')
    ]
    
    for idx, (metric, title) in enumerate(metrics, 1):
        plt.subplot(2, 3, idx)
        plt.plot(history[f'train_{metric}'], label=f'Train {title}')
        plt.plot(history[f'val_{metric}'], label=f'Val {title}')
        plt.title(f'{title} vs. Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(title)
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

def incremental_train(
    original_model_path: str,
    data_dir: str = 'processed_data_drift',
    save_dir: str = 'incremental_training_results_sangetisheng',
    num_epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-4,
    distill_alpha: float = 0.4
):
    """增量学习训练函数"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建梯度缩放器
    scaler = GradScaler()
    
    # 加载数据集
    dataset = IncrementalTrafficDataset(data_dir, 'train')
    
    # 获取漂移类别
    drift_classes = get_drift_classes()
    print(f"Drift classes: {drift_classes}")
    
    # 创建平衡的数据索引
    balanced_indices = create_balanced_indices(dataset)
    np.random.shuffle(balanced_indices)
    
    # 划分训练集和验证集 (90:10)
    train_size = int(0.9 * len(balanced_indices))
    train_indices = balanced_indices[:train_size]
    val_indices = balanced_indices[train_size:]
    
    # 创建数据加载器
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 加载模型
    model = TrafficMobileNetV3(num_classes=len(dataset.label_mapping))
    checkpoint = torch.load(original_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 添加特征注意力层
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        FeatureAttention(in_features),
        model.classifier
    )
    model = model.to(device)
    
    # 创建教师模型
    teacher_model = TrafficMobileNetV3(num_classes=len(dataset.label_mapping))
    teacher_model.load_state_dict(checkpoint['model_state_dict'])
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    
    # 创建损失函数
    criterion = WeightedDistillationLoss(
        num_classes=len(dataset.label_mapping),
        drift_classes=drift_classes,
        label_mapping=dataset.label_mapping
    )
    
    # 设置优化器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader)
    )
    
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
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")
        
        # 训练阶段
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc='Training')):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # 使用混合精度训练
            with autocast():
                student_output = model(data)
                with torch.no_grad():
                    teacher_output = teacher_model(data)
                loss = criterion(student_output, teacher_output, target, alpha=distill_alpha)
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 更新学习率
            scheduler.step()
            
            train_loss += loss.item()
            preds = student_output.argmax(dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(target.cpu().numpy())
        
        # 计算训练指标
        train_metrics = {
            'loss': train_loss / len(train_loader),
            'accuracy': accuracy_score(train_labels, train_preds),
            'precision': precision_score(train_labels, train_preds, average='weighted', zero_division=1),
            'recall': recall_score(train_labels, train_preds, average='weighted', zero_division=1),
            'f1': f1_score(train_labels, train_preds, average='weighted', zero_division=1)
        }
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Validation'):
                data, target = data.to(device), target.to(device)
                with autocast():
                    student_output = model(data)
                    teacher_output = teacher_model(data)
                    loss = criterion(student_output, teacher_output, target, alpha=distill_alpha)
                
                val_loss += loss.item()
                preds = student_output.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(target.cpu().numpy())
        
        # 计算验证指标
        val_metrics = {
            'loss': val_loss / len(val_loader),
            'accuracy': accuracy_score(val_labels, val_preds),
            'precision': precision_score(val_labels, val_preds, average='weighted', zero_division=1),
            'recall': recall_score(val_labels, val_preds, average='weighted', zero_division=1),
            'f1': f1_score(val_labels, val_preds, average='weighted', zero_division=1)
        }
        
        # 更新历史记录
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        # 更新历史记录（续）
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
        plot_confusion_matrix(cm, list(dataset.label_mapping.keys()), 
                            os.path.join(epoch_dir, 'confusion_matrix.png'))
        
        # 保存指标
        metrics = {
            'train': train_metrics,
            'val': val_metrics,
            'learning_rate': current_lr
        }
        with open(os.path.join(epoch_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # 保存最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': val_metrics,
                'focus_classes_metrics': {
                    class_name: {
                        'precision': precision_score(val_labels, val_preds, average='weighted', labels=[dataset.label_mapping[class_name]], zero_division=1),
                        'recall': recall_score(val_labels, val_preds, average='weighted', labels=[dataset.label_mapping[class_name]], zero_division=1),
                        'f1': f1_score(val_labels, val_preds, average='weighted', labels=[dataset.label_mapping[class_name]], zero_division=1)
                    }
                    for class_name in ['Heartbleed', 'Infiltration', 'PortScan', 'SSH-Patator']
                }
            }, os.path.join(save_dir, 'best_model.pth'))
            print("Saved new best model!")
        
        # 打印当前结果
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # 绘制训练曲线
        plot_metrics(history, os.path.join(save_dir, 'training_curves.png'))
    
    # 保存完整训练历史
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)

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
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # 配置参数
    config = {
        'original_model_path': 'model/best_model.pth',
        'data_dir': 'processed_data_drift',
        'save_dir': 'incremental_training_results_sangetisheng',
        'num_epochs': 50,
        'batch_size': 128,
        'learning_rate': 2e-4,
        'weight_decay': 1e-4,
        'distill_alpha': 0.4
    }
    
    print("Starting incremental learning with configurations:")
    for key, value in config.items():
        print(f"{key}: {value}")
    
    # 运行增量训练
    incremental_train(**config)