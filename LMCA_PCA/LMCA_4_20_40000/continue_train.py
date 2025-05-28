
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import json
import pandas as pd
from LMCA import LMCA
from training_utils import NetworkTrafficDataset, train_epoch, validate, plot_metrics

def continue_training(base_model_path, new_lr_schedule, save_dir='models_continue'):
    """基于已有模型继续训练"""
    # 配置参数
    data_dir = 'processed_data'
    num_classes = 11
    batch_size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建新的保存目录
    os.makedirs(save_dir, exist_ok=True)
    
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
    
    # 加载已有最佳模型
    print(f"Loading base model from {base_model_path}")
    model = LMCA(num_classes=num_classes).to(device)
    checkpoint = torch.load(base_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    criterion = nn.CrossEntropyLoss()
    best_val_acc = checkpoint['metrics']['val_accuracy']
    start_epoch = checkpoint['epoch']
    
    # 读取之前的训练历史
    try:
        prev_history = pd.read_csv(os.path.join(os.path.dirname(base_model_path), 
                                               'training_history.csv')).to_dict('records')
    except:
        prev_history = []
    
    metrics_history = prev_history
    total_epochs = start_epoch + sum(schedule['epochs'] for schedule in new_lr_schedule)
    
    print(f"Continuing training from epoch {start_epoch + 1}")
    print(f"Previous best validation accuracy: {best_val_acc:.4f}")
    current_epoch = start_epoch
    
    for schedule in new_lr_schedule:
        lr = schedule['lr']
        epochs = schedule['epochs']
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        print(f"\nTraining with learning rate: {lr} for {epochs} epochs")
        
        for epoch in range(epochs):
            current_epoch += 1
            print(f"\nEpoch {current_epoch}/{total_epochs}")
            
            # 创建当前epoch的保存目录
            epoch_dir = f'{save_dir}/epoch_{current_epoch}'
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
            
            # 保存当前epoch的指标和模型
            with open(f'{epoch_dir}/metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)
            
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
                }, f'{save_dir}/best_model.pth')
                print(f"Saved new best model with validation accuracy: {val_acc:.4f}")
            
            # 打印当前epoch的指标
            print(f"\nTrain - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
            print(f"Train - Precision: {train_prec:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
            print(f"Val - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
            print(f"Val - Precision: {val_prec:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
            
            # 绘制并保存训练曲线
            plot_metrics(metrics_history, save_dir)
    
    # 保存完整的训练历史
    pd.DataFrame(metrics_history).to_csv(f'{save_dir}/training_history.csv', index=False)
    
    print("\nContinue training completed!")
    print(f"Initial validation accuracy: {checkpoint['metrics']['val_accuracy']:.4f}")
    print(f"Final best validation accuracy: {best_val_acc:.4f}")
    print(f"All models and metrics have been saved in the '{save_dir}' directory")

if __name__ == '__main__':
    # 指定基础模型路径
    base_model_path = 'models_continue/best_model.pth'
    
    # 定义新的学习率调度
    new_lr_schedule = [
        {'epochs': 5, 'lr': 1e-5},   
        {'epochs': 5, 'lr': 1e-7},
        {'epochs': 10, 'lr': 1e-8}
    ]
    
    # 继续训练
    continue_training(base_model_path, new_lr_schedule, save_dir='models_continue_lr')
