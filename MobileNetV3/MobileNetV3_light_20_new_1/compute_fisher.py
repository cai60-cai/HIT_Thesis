import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 添加这一行
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from train import TrafficDataset, TrafficMobileNetV3

def plot_confusion_matrix(cm, classes, save_path):
    """绘制混淆矩阵"""
    fig = plt.figure(figsize=(10, 8))  # 修改这一行
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)  # 明确关闭图形

def evaluate_model(model, data_loader, device):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="Evaluating model"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)

def main():
    # 设置随机种子确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载原始模型
    original_model = TrafficMobileNetV3(num_classes=11)
    checkpoint_path = 'model/best_model.pth'
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    # 加载模型权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    original_model.load_state_dict(checkpoint['model_state_dict'])
    original_model = original_model.to(device)
    
    # 设置所有参数为可计算梯度
    for param in original_model.parameters():
        param.requires_grad = True
        
    print("Loaded original model successfully")
    
    # 数据路径
    data_dir = 'processed_data_org'
    print(f"\nUsing data directory: {data_dir}")
    
    try:
        # 加载训练和测试数据集
        train_dataset = TrafficDataset(data_dir, 'train')
        test_dataset = TrafficDataset(data_dir, 'test')
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        
        print(f"Loaded training dataset with {len(train_dataset)} samples")
        print(f"Loaded test dataset with {len(test_dataset)} samples")
        
        # 计算Fisher信息
        fisher_computer = FisherComputation(original_model, device)
        fisher_matrix, optimal_params = fisher_computer.compute_fisher_matrix(
            train_loader,
            num_batches=50
        )
        
        # 保存Fisher信息
        save_dict = {
            'fisher_matrix': fisher_matrix,
            'optimal_params': optimal_params,
            'model_state': original_model.state_dict()
        }
        
        save_path = 'fisher_matrix.pth'
        torch.save(save_dict, save_path)
        print(f"Saved Fisher matrix and optimal parameters to {save_path}")
        
        # 评估模型性能
        print("\nEvaluating model on test set...")
        predictions, true_labels = evaluate_model(original_model, test_loader, device)
        
        # 计算并保存混淆矩阵
        cm = confusion_matrix(true_labels, predictions)
        try:
            plot_confusion_matrix(cm, classes=range(11), save_path='confusion_matrix.png')
            print("Saved confusion matrix to confusion_matrix.png")
        except Exception as e:
            print(f"Error plotting confusion matrix: {e}")
            # 保存原始混淆矩阵数据
            np.save('confusion_matrix.npy', cm)
            print("Saved raw confusion matrix to confusion_matrix.npy")
        
        # 生成分类报告
        report = classification_report(true_labels, predictions)
        with open('classification_report.txt', 'w') as f:
            f.write(report)
        print("\nClassification Report:")
        print(report)
        
        # 输出Fisher信息统计
        print("\nFisher Information Statistics:")
        fisher_stats = {}
        for name, fisher_values in fisher_matrix.items():
            stats = {
                'mean': fisher_values.mean().item(),
                'std': fisher_values.std().item(),
                'max': fisher_values.max().item()
            }
            fisher_stats[name] = stats
            print(f"{name}:")
            print(f"  Mean: {stats['mean']:.6f}")
            print(f"  Std: {stats['std']:.6f}")
            print(f"  Max: {stats['max']:.6f}")
        
        # 保存所有统计信息
        stats_dict = {
            'fisher_stats': fisher_stats,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        torch.save(stats_dict, 'model_stats.pth')
        print("\nSaved all statistics to model_stats.pth")
            
    except Exception as e:
        print(f"Error during computation: {e}")
        import traceback
        traceback.print_exc()


class FisherComputation:
    def __init__(self, model, device='cuda:0'):
        self.model = model.to(device)
        self.device = device
        self.fisher = {}
        self.optpar = {}
        
        # 确保所有参数可以计算梯度
        for param in self.model.parameters():
            param.requires_grad = True
            
        # 初始化Fisher信息字典
        for name, param in model.named_parameters():
            self.fisher[name] = torch.zeros_like(param)
            self.optpar[name] = param.data.clone()

    def compute_fisher_batch(self, batch_data, batch_labels):
        """
        计算一个batch的Fisher信息
        """
        self.model.zero_grad()
        outputs = self.model(batch_data)
        
        # 对每个样本计算Fisher信息
        log_probs = F.log_softmax(outputs, dim=1)
        samples_probs = torch.gather(log_probs, 1, batch_labels.unsqueeze(1))
        
        # 计算每个参数的梯度
        for sample_prob in samples_probs:
            self.model.zero_grad()
            sample_prob.backward(retain_graph=True)
            
            # 累积Fisher信息
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher[name] += param.grad.data ** 2 / len(batch_data)

    def compute_fisher_matrix(self, data_loader, num_batches=100):
        """
        计算整个数据集的Fisher信息矩阵
        """
        self.model.eval()  # 设置为评估模式，但保持梯度计算
        processed_batches = 0
        
        print("Computing Fisher information matrix...")
        for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
            if batch_idx >= num_batches:
                break
                
            data, target = data.to(self.device), target.to(self.device)
            self.compute_fisher_batch(data, target)
            processed_batches += 1
        
        # normalize
        for name in self.fisher.keys():
            self.fisher[name] /= processed_batches
        
        return self.fisher, self.optpar

if __name__ == '__main__':
    main()




#import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import numpy as np
# import matplotlib.pyplot as plt
# import time
# from train import TrafficDataset, TrafficMobileNetV3

# class FisherAnalyzer:
#     def __init__(self, model, device='cuda:0'):
#         self.model = model.to(device)
#         self.device = device
#         self.batch_results = {}
        
#         # 确保所有参数可以计算梯度
#         for param in self.model.parameters():
#             param.requires_grad = True

#     def compute_fisher_for_batches(self, data_loader, max_batches=2000, step=100):
#         """计算不同batch数量下的Fisher信息"""
#         batch_numbers = range(step, max_batches + step, step)
#         fisher_stats = {
#             'mean_values': [],
#             'std_values': [],
#             'max_values': [],
#             'computation_times': []
#         }
        
#         for num_batches in tqdm(batch_numbers, desc="Testing different batch numbers"):
#             start_time = time.time()
            
#             # 计算Fisher信息
#             fisher = {}
#             for name, param in self.model.named_parameters():
#                 fisher[name] = torch.zeros_like(param)
            
#             for batch_idx, (data, target) in enumerate(data_loader):
#                 if batch_idx >= num_batches:
#                     break
                    
#                 data, target = data.to(self.device), target.to(self.device)
#                 self.model.zero_grad()
#                 outputs = self.model(data)
#                 log_probs = F.log_softmax(outputs, dim=1)
#                 samples_probs = torch.gather(log_probs, 1, target.unsqueeze(1))
                
#                 for sample_prob in samples_probs:
#                     self.model.zero_grad()
#                     sample_prob.backward(retain_graph=True)
                    
#                     for name, param in self.model.named_parameters():
#                         if param.grad is not None:
#                             fisher[name] += param.grad.data ** 2 / len(data)
            
#             # 计算统计量
#             all_values = torch.cat([f.flatten() for f in fisher.values()])
#             fisher_stats['mean_values'].append(all_values.mean().item())
#             fisher_stats['std_values'].append(all_values.std().item())
#             fisher_stats['max_values'].append(all_values.max().item())
#             fisher_stats['computation_times'].append(time.time() - start_time)
            
#         return batch_numbers, fisher_stats

#     def plot_results(self, batch_numbers, stats):
#         """绘制分析结果"""
#         plt.figure(figsize=(20, 15))
        
#         # 1. Fisher均值变化
#         plt.subplot(2, 2, 1)
#         plt.plot(batch_numbers, stats['mean_values'], marker='o')
#         plt.title('Fisher Information Mean vs. Number of Batches')
#         plt.xlabel('Number of Batches')
#         plt.ylabel('Mean Value')
#         plt.grid(True)
        
#         # 2. Fisher标准差变化
#         plt.subplot(2, 2, 2)
#         plt.plot(batch_numbers, stats['std_values'], marker='o', color='orange')
#         plt.title('Fisher Information Std vs. Number of Batches')
#         plt.xlabel('Number of Batches')
#         plt.ylabel('Standard Deviation')
#         plt.grid(True)
        
#         # 3. Fisher最大值变化
#         plt.subplot(2, 2, 3)
#         plt.plot(batch_numbers, stats['max_values'], marker='o', color='green')
#         plt.title('Fisher Information Max vs. Number of Batches')
#         plt.xlabel('Number of Batches')
#         plt.ylabel('Max Value')
#         plt.grid(True)
        
#         # 4. 计算时间
#         plt.subplot(2, 2, 4)
#         plt.plot(batch_numbers, stats['computation_times'], marker='o', color='red')
#         plt.title('Computation Time vs. Number of Batches')
#         plt.xlabel('Number of Batches')
#         plt.ylabel('Time (seconds)')
#         plt.grid(True)
        
#         plt.tight_layout()
#         plt.savefig('fisher_batch_analysis.png')
#         plt.close()

# def main():
#     # 设置设备
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     # 加载模型
#     model = TrafficMobileNetV3(num_classes=11)
#     checkpoint = torch.load('model/best_model.pth', map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
    
#     # 加载数据
#     data_dir = 'processed_data_org'
#     dataset = TrafficDataset(data_dir, 'train')
#     data_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
#     # 创建分析器并运行分析
#     analyzer = FisherAnalyzer(model, device)
#     batch_numbers, stats = analyzer.compute_fisher_for_batches(
#         data_loader,
#         max_batches=200,  # 最大测试到2000个batch
#         step=50           # 每100个batch为一个测试点
#     )
    
#     # 绘制结果
#     analyzer.plot_results(batch_numbers, stats)
    
#     # 输出建议的batch数量
#     mean_diffs = np.diff(stats['mean_values'])
#     stable_point = np.where(np.abs(mean_diffs) < np.mean(np.abs(mean_diffs)) * 0.1)[0]
#     if len(stable_point) > 0:
#         recommended_batches = (stable_point[0] + 1) * 100
#         print(f"\nRecommended number of batches: {recommended_batches}")
#         print(f"This will process approximately {recommended_batches * 256} samples")
    
#     print("\nAnalysis complete. Results saved to 'fisher_batch_analysis.png'")

# if __name__ == '__main__':
    main()