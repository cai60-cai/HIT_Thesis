import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

from dense_res_opt_mobilenetv3 import TrafficMobileNetV3

class NewClassTestDataset(Dataset):
    """新类别测试数据集加载器"""
    def __init__(self, data_dir):
        self.features = []
        batch_idx = 0
        while True:
            batch_path = f"{data_dir}/train_features_batch_{batch_idx}.npy"
            if not os.path.exists(batch_path):
                break
            self.features.append(np.load(batch_path))
            batch_idx += 1
            
        self.features = np.concatenate(self.features, axis=0)
        self.labels = np.load(f"{data_dir}/train_labels.npy")
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx]).unsqueeze(0)
        label = torch.LongTensor([self.labels[idx]])[0]
        return feature, label

class ThresholdAnalyzer:
    def __init__(self, model_path, device='cuda:1'):
        self.device = device
        
        # 加载模型
        checkpoint = torch.load(model_path)
        self.model = TrafficMobileNetV3(num_classes=11).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 待分析的百分位数
        self.percentiles = [95, 96, 97, 98, 99]
    
    def calculate_metrics(self, softmax_probs):
        """计算更多的统计指标"""
        metrics = defaultdict(list)
        
        # 获取top1, top2, top3概率
        top3_values, _ = torch.topk(softmax_probs, k=3, dim=1)
        
        # 基础概率
        max_probs, pred_classes = torch.max(softmax_probs, dim=1)
        metrics['max_probs'] = max_probs
        
        metrics['top1_prob'] = top3_values[:, 0]
        metrics['top2_prob'] = top3_values[:, 1]
        metrics['top3_prob'] = top3_values[:, 2]
        metrics['top3_sum'] = torch.sum(top3_values, dim=1)
        
        # 概率间隔
        metrics['margin_1_2'] = top3_values[:, 0] - top3_values[:, 1]  # 第1和第2的差
        metrics['margin_2_3'] = top3_values[:, 1] - top3_values[:, 2]  # 第2和第3的差
        
        # 定义 prob_diff 为 top1_prob 与 top2_prob 的差值
        metrics['prob_diff'] = metrics['top1_prob'] - metrics['top2_prob']
        
        # 熵和归一化熵
        entropy = -torch.sum(softmax_probs * torch.log(softmax_probs + 1e-10), dim=1)
        metrics['entropy'] = entropy
        metrics['normalized_entropy'] = entropy / np.log(softmax_probs.size(1))  # 归一化熵
        
        # KL散度(与均匀分布比较)
        uniform_dist = torch.ones_like(softmax_probs) / softmax_probs.size(1)
        kl_div = torch.sum(softmax_probs * torch.log(softmax_probs / uniform_dist + 1e-10), dim=1)
        metrics['kl_divergence'] = kl_div
        
        # 方差和标准差
        variance = torch.var(softmax_probs, dim=1)
        metrics['variance'] = variance
        metrics['std_dev'] = torch.sqrt(variance)
        
        # 基尼系数
        sorted_probs, _ = torch.sort(softmax_probs, dim=1)
        n = softmax_probs.size(1)
        indices = torch.arange(1, n + 1, device=softmax_probs.device)
        gini = 2 * torch.sum(sorted_probs * indices.view(1, -1), dim=1) / (n * torch.sum(sorted_probs, dim=1)) - (n + 1) / n
        metrics['gini'] = gini
        
        # 综合特征
        metrics['entropy_margin_ratio'] = entropy / (metrics['margin_1_2'] + 1e-10)  # 熵与margin的比率
        metrics['top3_entropy_ratio'] = metrics['top3_sum'] / (entropy + 1e-10)  # top3和与熵的比率
        
        return metrics

    def analyze_thresholds(self, data_loader):
        """分析所有指标的分布"""
        stats = defaultdict(list)
        
        with torch.no_grad():
            for data, labels in tqdm(data_loader, desc="Analyzing training data statistics"):
                data = data.to(self.device)
                outputs = self.model(data)
                softmax_probs = F.softmax(outputs, dim=1)
                
                # 计算所有指标
                metrics = self.calculate_metrics(softmax_probs)
                
                # 获取正确预测的mask
                pred_classes = torch.argmax(softmax_probs, dim=1)
                correct_mask = pred_classes == labels.to(self.device)
                
                # 收集正确预测样本的统计量
                for metric_name, values in metrics.items():
                    stats[metric_name].extend(values[correct_mask].cpu().numpy())
        
        # 计算每个统计量的阈值和分布
        thresholds = self._calculate_percentile_thresholds(stats)
        
        return thresholds

    def _calculate_percentile_thresholds(self, stats):
        """计算所有指标的阈值和分布信息"""
        thresholds = {}
        
        for metric, values in stats.items():
            values = np.array(values)
            
            # 根据指标类型选择百分位方向
            # 对于希望越大越好的指标，使用较小的百分位数作为阈值(例如max_probs)
            # 对于希望越小越好的指标(如熵?), 可以视情况调整方向。
            # 这里假设和原代码保持一致的逻辑。
            if metric in ['max_probs', 'top1_prob', 'top2_prob', 'top3_prob', 
                          'top3_sum', 'margin_1_2', 'margin_2_3', 'gini', 'prob_diff']:
                percentile_values = np.percentile(values, self.percentiles)
            else:
                percentile_values = np.percentile(values, [100 - p for p in self.percentiles])
            
            thresholds[metric] = {
                f'p{p}': float(val)
                for p, val in zip(self.percentiles, percentile_values)
            }
            
            # 添加基本统计信息
            thresholds[metric].update({
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            })
        
        return thresholds

def main():
    """主函数"""
    # 配置
    data_dir = 'processed_data_org'  # 原始训练数据目录
    model_path = 'model/best_model.pth'
    output_dir = 'threshold_analysis_results'
    batch_size = 256
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载训练数据
    dataset = NewClassTestDataset(data_dir)  # 使用训练数据
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 分析阈值
    analyzer = ThresholdAnalyzer(model_path)
    thresholds = analyzer.analyze_thresholds(data_loader)
    
    # 打印分析结果
    print("\nThreshold Analysis Results")
    print("-" * 50)
    
    for metric, stats in thresholds.items():
        print(f"\n{metric.upper()} Statistics:")
        print(f"Mean: {stats['mean']:.4f}")
        print(f"Std: {stats['std']:.4f}")
        print("\nPercentile Thresholds:")
        for p in analyzer.percentiles:
            print(f"{p}th percentile: {stats[f'p{p}']:.4f}")
    
    # 保存结果
    with open(os.path.join(output_dir, 'threshold_analysis.json'), 'w') as f:
        json.dump(thresholds, f, indent=4)
    
    print(f"\nResults saved to {output_dir}")
    
    # 推荐的阈值组合
    # 现在prob_diff存在, 可以正常访问
    recommended_thresholds = {
        'max_probs': thresholds['max_probs']['p95'],
        'entropy': thresholds['entropy']['p95'],
        'prob_diff': thresholds['prob_diff']['p95'],
        'variance': thresholds['variance']['p95']
    }
    
    print("\nRecommended Thresholds:")
    for metric, value in recommended_thresholds.items():
        print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    main()
