import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import json
import copy
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import pandas as pd

from dense_res_opt_mobilenetv3 import TrafficMobileNetV3

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

class IncrementalKnowledgeDistillation:
    def __init__(
        self, 
        original_model_path, 
        num_new_classes=2, 
        temperature=2.0, 
        alpha=0.5
    ):
        # 设备配置
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 加载原始模型
        checkpoint = torch.load(original_model_path)
        self.original_model = TrafficMobileNetV3(num_classes=11)
        self.original_model.load_state_dict(checkpoint['model_state_dict'])
        self.original_model.to(self.device)
        self.original_model.eval()  # 固定原模型
        
        # 创建学生模型
        self.student_model = copy.deepcopy(self.original_model)
        
        # 替换分类器最后一层
        original_classifier = list(self.student_model.classifier)
        original_classifier[-1] = nn.Linear(
            original_classifier[-1].in_features, 
            11 + num_new_classes
        )
        self.student_model.classifier = nn.Sequential(*original_classifier).to(self.device)
        
        # 蒸馏参数
        self.temperature = temperature
        self.alpha = alpha
        
        # 训练配置
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()

    def soft_target_loss(self, student_outputs, teacher_outputs):
        """知识蒸馏软标签损失"""
        # 仅使用原始类别的输出进行知识蒸馏
        student_original_outputs = student_outputs[:, :11]
        
        soft_loss = F.kl_div(
            F.log_softmax(student_original_outputs / self.temperature, dim=1),
            F.softmax(teacher_outputs / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        return soft_loss

    def train_incremental(
        self, 
        train_loader, 
        val_loader, 
        save_dir='incremental_results',
        num_epochs=30, 
        learning_rate=1e-4,
    ):
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 优化器和学习率调度
        self.optimizer = optim.Adam(
            self.student_model.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-5
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs
        )

        # 训练历史记录
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
            self.student_model.train()
            train_loss = 0
            train_preds, train_labels = [], []
            
            for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc='Training')):
                data, target = data.to(self.device), target.to(self.device)
                
                # 获取教师模型软标签
                with torch.no_grad():
                    teacher_outputs = self.original_model(data)

                # 学生模型预测
                student_outputs = self.student_model(data)
                
                # 组合损失
                hard_loss = self.criterion(student_outputs, target)
                soft_loss = self.soft_target_loss(student_outputs, teacher_outputs)
                loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                preds = student_outputs.argmax(dim=1).cpu().numpy()
                train_preds.extend(preds)
                train_labels.extend(target.cpu().numpy())
            
            train_metrics = self.compute_metrics(train_labels, train_preds)
            train_metrics['loss'] = train_loss / len(train_loader)
            
            # 验证阶段
            self.student_model.eval()
            val_loss = 0
            val_preds, val_labels = [], []
            
            with torch.no_grad():
                for data, target in tqdm(val_loader, desc='Validation'):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    teacher_outputs = self.original_model(data)
                    student_outputs = self.student_model(data)
                    
                    hard_loss = self.criterion(student_outputs, target)
                    soft_loss = self.soft_target_loss(student_outputs, teacher_outputs)
                    loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
                    
                    val_loss += loss.item()
                    preds = student_outputs.argmax(dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_labels.extend(target.cpu().numpy())
            
            val_metrics = self.compute_metrics(val_labels, val_preds)
            val_metrics['loss'] = val_loss / len(val_loader)
            
            # 学习率调整
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            
            # 更新历史记录 (与原训练代码保持一致)
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
            
            # 打印结果
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            
            # 绘制训练曲线 (与原训练代码保持一致)
            plot_metrics(history, os.path.join(save_dir, 'training_curves.png'))
            
            # 保存最佳模型 (与原训练代码保持一致)
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.student_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'metrics': val_metrics
                }, os.path.join(save_dir, 'best_incremental_model.pth'))
                print("Saved new best incremental model!")
        
        return self.student_model

    def compute_metrics(self, labels, preds):
        """计算评估指标"""
        return {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, average='weighted'),
            'recall': recall_score(labels, preds, average='weighted'),
            'f1': f1_score(labels, preds, average='weighted')
        }

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

def incremental_train():
    # 配置参数
    original_model_path = 'model/best_model.pth'
    data_dir = 'processed_data_incremental'
    save_dir = 'incremental_results'
    
    # 加载数据
    dataset = TrafficDataset(data_dir, 'train')
    
    # 加载标签映射
    label_mapping = pd.read_csv(f"{data_dir}/label_mapping.csv", index_col=0)
    
    # 划分训练集和验证集 (95:5)
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # 初始化知识蒸馏增量学习
    incremental_trainer = IncrementalKnowledgeDistillation(
        original_model_path, 
        num_new_classes=2,  # 新增的类别数
        temperature=2.0,
        alpha=0.5
    )
    
    # 训练并保存模型
    incremental_trainer.train_incremental(
        train_loader, 
        val_loader, 
        save_dir=save_dir
    )

if __name__ == '__main__':
    incremental_train()