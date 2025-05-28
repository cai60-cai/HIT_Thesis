import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 适用于无显示环境
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# 假设 train.py 中定义了 TrafficDataset 和 TrafficMobileNetV3
from train import TrafficDataset, TrafficMobileNetV3

# 定义 WebAttackFeatureExtractor
class WebAttackFeatureExtractor(nn.Module):
    """专门用于Web攻击特征提取的模块"""
    def __init__(self, in_channels):
        super(WebAttackFeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.conv(x)

# 创建必要的目录
def create_directories(base_dir='outputs'):
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    confusion_matrix_dir = os.path.join(base_dir, 'confusion_matrices')
    reports_dir = os.path.join(base_dir, 'reports')
    logs_dir = os.path.join(base_dir, 'logs')
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(confusion_matrix_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    return {
        'checkpoint_dir': checkpoint_dir,
        'confusion_matrix_dir': confusion_matrix_dir,
        'reports_dir': reports_dir,
        'logs_dir': logs_dir
    }

def plot_confusion_matrix(cm, classes, save_path):
    """绘制并保存混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model, web_attack_extractor, data_loader, device):
    """评估模型性能，返回预测和真实标签"""
    model.eval()
    web_attack_extractor.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target, _ in tqdm(data_loader, desc="Evaluating model"):
            data, target = data.to(device), target.to(device)
            features = model.get_features(data)
            web_features = web_attack_extractor(data)
            combined_features = torch.cat([features, web_features.mean((2, 3))], dim=1)
            output = model.classifier(combined_features)
            preds = output.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)

def main():
    # 创建保存目录
    dirs = create_directories(base_dir='outputs')
    
    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 初始化模型
    num_classes = 12  # 根据您的实际情况调整
    model = TrafficMobileNetV3(num_classes=num_classes).to(device)
    
    # 初始化 WebAttackFeatureExtractor
    web_attack_extractor = WebAttackFeatureExtractor(in_channels=1).to(device)
    
    # 加载训练好的模型权重
    checkpoint_path = os.path.join(dirs['checkpoint_dir'], 'best_incremental_model.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载 model state_dict
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # 加载 web_attack_extractor state_dict
    if 'web_attack_extractor_state_dict' in checkpoint:
        web_attack_extractor.load_state_dict(checkpoint['web_attack_extractor_state_dict'], strict=False)
    else:
        print("Warning: 'web_attack_extractor_state_dict' not found in checkpoint.")
    
    # 设置模型为评估模式
    model.eval()
    web_attack_extractor.eval()
    print("Loaded trained model and WebAttackFeatureExtractor successfully")
    
    # 加载测试数据集
    test_dataset = TrafficDataset(data_dir='processed_data_last_5_3', prefix='test', is_new_class=True)  # 根据需要调整
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Loaded test dataset with {len(test_dataset)} samples")
    
    # 评估模型
    print("\nEvaluating model on test set...")
    predictions, true_labels = evaluate_model(model, web_attack_extractor, test_loader, device)
    
    # 生成混淆矩阵
    cm = confusion_matrix(true_labels, predictions)
    class_names = [str(i) for i in range(num_classes)]  # 根据实际类别名称调整
    plot_confusion_matrix(cm, classes=class_names, save_path=os.path.join(dirs['confusion_matrix_dir'], 'confusion_matrix_test.png'))
    print(f"Saved confusion matrix to {os.path.join(dirs['confusion_matrix_dir'], 'confusion_matrix_test.png')}")
    
    # 生成分类报告
    report = classification_report(true_labels, predictions, digits=4)
    report_path = os.path.join(dirs['reports_dir'], 'classification_report_test.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved classification report to {report_path}")
    print("\nClassification Report:")
    print(report)
    
    # 计算并打印F1分数
    f1 = f1_score(true_labels, predictions, average='weighted')
    print(f"Weighted F1 Score on Test Set: {f1:.4f}")
    
    # 保存所有统计信息
    stats_dict = {
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'weighted_f1_score': f1
    }
    stats_save_path = os.path.join(dirs['reports_dir'], 'test_stats.pth')
    torch.save(stats_dict, stats_save_path)
    print(f"Saved all statistics to {stats_save_path}")

if __name__ == '__main__':
    main()
