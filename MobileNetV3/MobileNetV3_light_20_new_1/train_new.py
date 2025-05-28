import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import os
import json
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from train import plot_metrics, plot_confusion_matrix  # 从你的原始代码导入
from train import TrafficDataset  # 从你的原始代码导入
import torch.nn.functional as F
# from IncrementalMobileNetV3 import IncrementalTrafficMobileNetV3
from enhanced_model import IncrementalTrafficMobileNetV3


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # 确保 alpha 是 tensor
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
    def __init__(self, data_dir, prefix, is_new_class=False):
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
        
        # 如果是新类别，调整标签
        if is_new_class:
            self.labels = self.labels + 11  # 假设原来有11个类别
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx]).unsqueeze(0)
        label = torch.LongTensor([self.labels[idx]])[0]
        return feature, label

def train_incremental():
    # 配置参数
    old_data_dir = 'processed_data_org'
    new_data_dir = 'processed_data_incremental_last_5_3'  # 新类别数据目录
    save_dir = 'training_results_incremental_buyuohua_webmokuai'
    old_model_path = 'model/best_model.pth'
    num_epochs = 30
    batch_size = 256
    learning_rate = 1e-4  # 降低学习率
    weight_decay = 1e-5
    temperature = 2.0  # 知识蒸馏温度
    alpha = 0.5  # 知识蒸馏损失权重
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    old_train_dataset = TrafficDataset(old_data_dir, 'train')
    old_val_dataset = TrafficDataset(old_data_dir, 'test')
    new_train_dataset = IncrementalTrafficDataset(new_data_dir, 'train', is_new_class=True)
    new_val_dataset = IncrementalTrafficDataset(new_data_dir, 'test', is_new_class=True)
    
    # 合并数据集
    train_dataset = ConcatDataset([old_train_dataset, new_train_dataset])
    val_dataset = ConcatDataset([old_val_dataset, new_val_dataset])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 初始化模型
    model = IncrementalTrafficMobileNetV3(old_model_path, old_num_classes=11, new_num_classes=12).to(device)
    
    # # 定义类别权重，特别增加 PortScan(index=9) 和 NewClass(index=11) 的权重
    # class_weights = torch.ones(12, device=device)  # 12个类别的权重数组
    # class_weights[9] = 2.0  # PortScan的权重
    # class_weights[11] = 2.0  # NewClass的权重

    class_weights = torch.ones(12)
    class_weights[9] = 1.0  # PortScan的权重
    class_weights[11] = 1.0  # NewClass的权重

      # 损失函数
    criterion = FocalLoss(gamma=2, alpha=class_weights).to(device)
    distillation_criterion = nn.KLDivLoss(reduction='batchmean')
    
    # 使用Focal Loss
    # criterion = FocalLoss(gamma=2, alpha=alpha)
    # distillation_criterion = nn.KLDivLoss(reduction='batchmean')
    
    # # 损失函数
    # criterion = nn.CrossEntropyLoss()
    # distillation_criterion = nn.KLDivLoss(reduction='batchmean')
    
    # 优化器 - 只优化新添加的层
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
        # 分类损失
        # cls_loss = criterion(outputs, labels)
        focal_loss = criterion(outputs, labels)
        if old_outputs is not None:
            old_probs = F.softmax(old_outputs / temperature, dim=1)
            new_log_probs = F.log_softmax(outputs[:, :11] / temperature, dim=1)
            distill_loss = distillation_criterion(new_log_probs, old_probs)
            return focal_loss + alpha * (temperature ** 2) * distill_loss
            
        return focal_loss
        # 如果有旧模型输出，添加知识蒸馏损失
        # if old_outputs is not None:
        #     # 只对旧类别进行蒸馏
        #     old_probs = F.softmax(old_outputs / temperature, dim=1)
        #     new_log_probs = F.log_softmax(outputs[:, :11] / temperature, dim=1)
        #     distill_loss = distillation_criterion(new_log_probs, old_probs)

        #     # # 增加PortScan和NewClass之间的对抗损失
        #     # port_scan_new_loss = torch.mean(
        #     #     outputs[:, 9] * outputs[:, 11]  # 降低这两个类别同时激活的可能性
        #     # )
            
        #     # return cls_loss + alpha * distill_loss + 0.1 * port_scan_new_loss
        #     return focal_loss + alpha * (temperature ** 2) * distill_loss
        #     # return cls_loss + alpha * distill_loss
        
        # # return cls_loss
        # return focal_loss
    
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
        
         # 修改指标的记录方式
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
        epoch_dir = os.path.join(save_dir, f'epoch_{epoch+1}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        # 保存混淆矩阵
        cm = confusion_matrix(val_labels, val_preds)
        class_names = [str(i) for i in range(12)]  # 12个类别
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
                'metrics': val_metrics,
                'old_num_classes': model.old_num_classes,
                'new_num_classes': model.new_num_classes
            }, os.path.join(save_dir, 'best_model.pth'))
            print("Saved new best model!")
        
        # 打印当前结果
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # 绘制并保存训练曲线
        plot_metrics(history, os.path.join(save_dir, 'training_curves.png'))
    
    # 保存完整训练历史
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)

def evaluate_incremental_model(model_path, old_data_dir, new_data_dir, device=None):
    """评估增量学习模型的性能"""
    if device is None:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    model = IncrementalTrafficMobileNetV3(
        old_model_path=None,  # 不需要加载旧模型，因为状态已经在checkpoint中
        old_num_classes=checkpoint['old_num_classes'],
        new_num_classes=checkpoint['new_num_classes']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载测试数据
    old_test_dataset = TrafficDataset(old_data_dir, 'test')
    new_test_dataset = IncrementalTrafficDataset(new_data_dir, 'test', is_new_class=True)
    test_dataset = ConcatDataset([old_test_dataset, new_test_dataset])
    test_loader = DataLoader(test_dataset, batch_size=256)
    
    # 进行预测
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluating'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(target.cpu().numpy())
    
    # 计算指标
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted'),
        'recall': recall_score(all_labels, all_preds, average='weighted'),
        'f1': f1_score(all_labels, all_preds, average='weighted')
    }
    
    # 计算每个类别的指标
    class_metrics = {
        'precision': precision_score(all_labels, all_preds, average=None),
        'recall': recall_score(all_labels, all_preds, average=None),
        'f1': f1_score(all_labels, all_preds, average=None)
    }
    
    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    return metrics, class_metrics, cm

if __name__ == '__main__':
    # 训练模型
    train_incremental()
    
    # 评估模型
    model_path = 'training_results_incremental/best_model.pth'
    old_data_dir = 'processed_data_incremental_last_5_3'
    new_data_dir = 'new_class_processed_data'
    
    metrics, class_metrics, cm = evaluate_incremental_model(
        model_path, old_data_dir, new_data_dir
    )
    
    # 打印整体指标
    print("\nOverall Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # 打印每个类别的指标
    print("\nPer-class Metrics:")
    for i in range(len(class_metrics['f1'])):
        print(f"\nClass {i}:")
        for metric_name, values in class_metrics.items():
            print(f"{metric_name}: {values[i]:.4f}")
    
    # 保存评估结果
    results = {
        'overall_metrics': metrics,
        'class_metrics': {
            k: v.tolist() for k, v in class_metrics.items()
        },
        'confusion_matrix': cm.tolist()
    }
    
    os.makedirs('evaluation_results', exist_ok=True)
    with open('evaluation_results/incremental_evaluation.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # 绘制混淆矩阵
    class_names = [str(i) for i in range(12)]
    plot_confusion_matrix(
        cm, class_names, 
        'evaluation_results/final_confusion_matrix.png'
    )