import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from dense_res_opt_mobilenetv3 import TrafficMobileNetV3
from IncrementalTraining import (
    TrafficDataset, 
    create_directories
)

def create_adaptive_classifier(input_dim, num_classes=12):
    """动态创建分类器"""
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(64, num_classes)
    )

def test_incremental_model():
    # 设备配置
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # 创建保存目录
    dirs = create_directories(base_dir='test_outputs')
    
    # 加载测试数据集
    try:
        old_test_dataset = TrafficDataset('processed_data_org', 'test', is_new_class=False)
        new_test_dataset = TrafficDataset('processed_data_incremental_last_5_3', 'test', is_new_class=True)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Checking if datasets exist...")
        print("Old dataset exists:", os.path.exists('processed_data_org'))
        print("Incremental dataset exists:", os.path.exists('processed_data_incremental_last_5_3'))
        raise
    
    # 合并测试数据集
    combined_test_dataset = torch.utils.data.ConcatDataset([old_test_dataset, new_test_dataset])
    
    # 创建测试数据加载器
    test_loader = torch.utils.data.DataLoader(
        combined_test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 初始化模型
    model = TrafficMobileNetV3(num_classes=12).to(device)
    
    # 加载最佳增量模型
    checkpoint_path = 'outputs/checkpoints/best_incremental_model.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 动态确定特征维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 5, 5).to(device)
            features = model.get_features(dummy_input)
            feature_dim = features.shape[1]
        
        # 创建新的分类器
        model.classifier = create_adaptive_classifier(feature_dim, num_classes=12).to(device)
        
        # 加载模型权重
        model_state_dict = checkpoint['model_state_dict']
        current_state_dict = model.state_dict()
        
        # 只加载形状匹配的权重
        for name, param in model_state_dict.items():
            if name in current_state_dict and param.shape == current_state_dict[name].shape:
                current_state_dict[name].copy_(param)
            elif 'classifier' in name:
                print(f"Skipping {name} due to shape mismatch")
    else:
        raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found.")
    
    # 设置模型为评估模式
    model.eval()
    
    # 存储预测结果
    all_preds = []
    all_labels = []
    
    # 禁用梯度计算
    with torch.no_grad():
        for data, target, _ in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)
            
            # 特征提取
            features = model.get_features(data)
            
            # 分类
            output = model.classifier(features)
            
            # 收集预测结果
            preds = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(target.cpu().numpy())
    
    # 计算整体指标
    overall_f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"\nOverall Test F1 Score: {overall_f1:.4f}")
    
    # 生成并保存混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Test Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_save_path = os.path.join(dirs['confusion_matrix_dir'], 'test_confusion_matrix.png')
    plt.savefig(cm_save_path)
    plt.close()
    print(f"Saved test confusion matrix to {cm_save_path}")
    
    # 详细分类报告
    test_report = classification_report(all_labels, all_preds, digits=4)
    report_save_path = os.path.join(dirs['reports_dir'], 'test_classification_report.txt')
    with open(report_save_path, 'w') as f:
        f.write(test_report)
    print(f"Saved test classification report to {report_save_path}")
    print("\nDetailed Classification Report:")
    print(test_report)
    
    # 按类别分析性能
    unique_labels = np.unique(all_labels)
    print("\nPer-Class Performance:")
    for label in unique_labels:
        class_mask = np.array(all_labels) == label
        class_preds = np.array(all_preds)[class_mask]
        class_true = np.array(all_labels)[class_mask]
        
        class_f1 = f1_score(class_true, class_preds, average='micro')
        print(f"Class {label} - F1 Score: {class_f1:.4f}")
    
    # 特别关注增量类别的性能
    new_class_indices = [11, 12]  # 假设新类别的索引是11和12
    new_class_mask = np.isin(all_labels, new_class_indices)
    
    if new_class_mask.sum() > 0:
        new_class_labels = np.array(all_labels)[new_class_mask]
        new_class_preds = np.array(all_preds)[new_class_mask]
        
        new_class_report = classification_report(
            new_class_labels, 
            new_class_preds, 
            digits=4
        )
        
        new_class_report_path = os.path.join(dirs['reports_dir'], 'new_class_test_report.txt')
        with open(new_class_report_path, 'w') as f:
            f.write(new_class_report)
        
        print("\nNew Classes Performance:")
        print(new_class_report)
    else:
        print("\nNo new class samples found in the test set.")

if __name__ == '__main__':
    test_incremental_model()


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import os
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, classification_report, f1_score

# from dense_res_opt_mobilenetv3 import TrafficMobileNetV3
# from IncrementalTraining import (
#     TrafficDataset, 
#     WebAttackFeatureExtractor,
#     create_directories
# )

# def create_adaptive_classifier(input_dim, num_classes=12):
#     """动态创建分类器"""
#     return nn.Sequential(
#         nn.Linear(input_dim, 128),
#         nn.BatchNorm1d(128),
#         nn.ReLU(inplace=True),
#         nn.Dropout(0.2),
#         nn.Linear(128, 64),
#         nn.BatchNorm1d(64),
#         nn.ReLU(inplace=True),
#         nn.Dropout(0.2),
#         nn.Linear(64, num_classes)
#     )

# def test_incremental_model():
#     # 设备配置
#     device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
#     # 创建保存目录
#     dirs = create_directories(base_dir='test_outputs')
    
#     # 加载测试数据集
#     try:
#         old_test_dataset = TrafficDataset('processed_data_org', 'test', is_new_class=False)
#         new_test_dataset = TrafficDataset('processed_data_incremental_last_5_3', 'test', is_new_class=True)
#     except Exception as e:
#         print(f"Error loading datasets: {e}")
#         print("Checking if datasets exist...")
#         print("Old dataset exists:", os.path.exists('processed_data_org'))
#         print("Incremental dataset exists:", os.path.exists('processed_data_incremental_last_5_3'))
#         raise
    
#     # 合并测试数据集
#     combined_test_dataset = torch.utils.data.ConcatDataset([old_test_dataset, new_test_dataset])
    
#     # 创建测试数据加载器
#     test_loader = torch.utils.data.DataLoader(
#         combined_test_dataset,
#         batch_size=256,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True
#     )
    
#     # 初始化模型
#     model = TrafficMobileNetV3(num_classes=12).to(device)
#     web_attack_extractor = WebAttackFeatureExtractor(in_channels=1).to(device)
    
#     # 加载最佳增量模型
#     checkpoint_path = 'outputs/checkpoints/best_incremental_model.pth'
#     if os.path.exists(checkpoint_path):
#         checkpoint = torch.load(checkpoint_path, map_location=device)
        
#         # 处理特征提取器的状态字典
#         web_attack_extractor_state_dict = checkpoint.get('web_attack_extractor_state_dict', {})
#         web_attack_extractor.load_state_dict(web_attack_extractor_state_dict)
        
#         # 动态确定特征维度
#         with torch.no_grad():
#             dummy_input = torch.randn(1, 1, 5, 5).to(device)
#             features = model.get_features(dummy_input)
#             web_features = web_attack_extractor(dummy_input)
#             combined_features_dim = features.shape[1] + web_features.mean((2, 3)).shape[1]
        
#         # 创建新的分类器
#         model.classifier = create_adaptive_classifier(combined_features_dim, num_classes=12).to(device)
        
#         # 加载模型权重
#         model_state_dict = checkpoint['model_state_dict']
#         current_state_dict = model.state_dict()
        
#         # 只加载形状匹配的权重
#         for name, param in model_state_dict.items():
#             if name in current_state_dict and param.shape == current_state_dict[name].shape:
#                 current_state_dict[name].copy_(param)
#             elif 'classifier' in name:
#                 print(f"Skipping {name} due to shape mismatch")
#     else:
#         raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found.")
    
#     # 设置模型为评估模式
#     model.eval()
#     web_attack_extractor.eval()
    
#     # 存储预测结果
#     all_preds = []
#     all_labels = []
    
#     # 禁用梯度计算
#     with torch.no_grad():
#         for data, target, _ in tqdm(test_loader, desc='Testing'):
#             data, target = data.to(device), target.to(device)
            
#             # 特征提取
#             features = model.get_features(data)
#             web_features = web_attack_extractor(data)
#             combined_features = torch.cat([features, web_features.mean((2, 3))], dim=1)
            
#             # 分类
#             output = model.classifier(combined_features)
            
#             # 收集预测结果
#             preds = output.argmax(dim=1).cpu().numpy()
#             all_preds.extend(preds)
#             all_labels.extend(target.cpu().numpy())
    
#     # 计算整体指标
#     overall_f1 = f1_score(all_labels, all_preds, average='weighted')
#     print(f"\nOverall Test F1 Score: {overall_f1:.4f}")
    
#     # 生成并保存混淆矩阵
#     cm = confusion_matrix(all_labels, all_preds)
#     plt.figure(figsize=(15, 12))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.title('Test Confusion Matrix')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.tight_layout()
#     cm_save_path = os.path.join(dirs['confusion_matrix_dir'], 'test_confusion_matrix.png')
#     plt.savefig(cm_save_path)
#     plt.close()
#     print(f"Saved test confusion matrix to {cm_save_path}")
    
#     # 详细分类报告
#     test_report = classification_report(all_labels, all_preds, digits=4)
#     report_save_path = os.path.join(dirs['reports_dir'], 'test_classification_report.txt')
#     with open(report_save_path, 'w') as f:
#         f.write(test_report)
#     print(f"Saved test classification report to {report_save_path}")
#     print("\nDetailed Classification Report:")
#     print(test_report)
    
#     # 按类别分析性能
#     unique_labels = np.unique(all_labels)
#     print("\nPer-Class Performance:")
#     for label in unique_labels:
#         class_mask = np.array(all_labels) == label
#         class_preds = np.array(all_preds)[class_mask]
#         class_true = np.array(all_labels)[class_mask]
        
#         class_f1 = f1_score(class_true, class_preds, average='micro')
#         print(f"Class {label} - F1 Score: {class_f1:.4f}")
    
#     # 特别关注增量类别的性能
#     new_class_indices = [11, 12]  # 假设新类别的索引是11和12
#     new_class_mask = np.isin(all_labels, new_class_indices)
    
#     if new_class_mask.sum() > 0:
#         new_class_labels = np.array(all_labels)[new_class_mask]
#         new_class_preds = np.array(all_preds)[new_class_mask]
        
#         new_class_report = classification_report(
#             new_class_labels, 
#             new_class_preds, 
#             digits=4
#         )
        
#         new_class_report_path = os.path.join(dirs['reports_dir'], 'new_class_test_report.txt')
#         with open(new_class_report_path, 'w') as f:
#             f.write(new_class_report)
        
#         print("\nNew Classes Performance:")
#         print(new_class_report)
#     else:
#         print("\nNo new class samples found in the test set.")

# if __name__ == '__main__':
#     test_incremental_model()