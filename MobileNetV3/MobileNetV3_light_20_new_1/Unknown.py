import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
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
import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from dense_res_opt_mobilenetv3 import TrafficMobileNetV3

class UnknownDetector:
    def __init__(self, model_path, threshold=0.8421, device='cuda:1'):
        self.threshold = threshold
        self.device = device
        
        # 加载模型
        checkpoint = torch.load(model_path)
        self.model = TrafficMobileNetV3(num_classes=11).to(device)  # 原始11类
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def analyze_unknown(self, data_loader):
        """分析新类别的检测效果，并按类别统计未知样本"""
        all_confidences = []
        all_predictions = []
        all_labels = []
        unknown_count_by_class = {}
        total_count_by_class = {}
        
        with torch.no_grad():
            for data, labels in tqdm(data_loader, desc="Analyzing unknown detection"):
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                # 获取模型预测
                outputs = self.model(data)
                softmax_probs = F.softmax(outputs, dim=1)
                max_probs, pred_classes = torch.max(softmax_probs, dim=1)
                
                # 记录每个样本的信息
                all_confidences.extend(max_probs.cpu().numpy())
                all_predictions.extend(pred_classes.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # 按类别统计
                for label, prob, pred_class in zip(labels.cpu().numpy(), 
                                                   max_probs.cpu().numpy(), 
                                                   pred_classes.cpu().numpy()):
                    # 初始化计数器
                    if label not in total_count_by_class:
                        total_count_by_class[label] = 0
                        unknown_count_by_class[label] = 0
                    
                    total_count_by_class[label] += 1
                    
                    # 如果置信度低于阈值，计为未知
                    if prob < self.threshold:
                        unknown_count_by_class[label] += 1
        
        # 转换为numpy数组
        all_confidences = np.array(all_confidences)
        
        # 计算统计信息
        unknown_rate_by_class = {
            label: unknown_count_by_class[label] / total_count_by_class[label] 
            for label in total_count_by_class
        }
        
        results = {
            'total_samples': int(sum(total_count_by_class.values())),
            'detected_unknown_by_class': {int(k): int(v) for k, v in unknown_count_by_class.items()},
            'total_samples_by_class': {int(k): int(v) for k, v in total_count_by_class.items()},
            'unknown_rate_by_class': {int(k): float(v*100) for k, v in unknown_rate_by_class.items()},
            'overall_detection_rate': float(sum(unknown_count_by_class.values()) / sum(total_count_by_class.values())),
            'confidence_stats': {
                'mean': float(np.mean(all_confidences)),
                'max': float(np.max(all_confidences)),
                'min': float(np.min(all_confidences)),
                'std': float(np.std(all_confidences)),
                'median': float(np.median(all_confidences)),
            }
        }
        
        return results

def main():
    # 配置
    data_dir = 'processed_data_incremental_2'  # 新类别数据目录
    model_path = 'model/best_model.pth'
    output_dir = 'unknown_detection_results_incremental_2'
    threshold = 0.9996299743652344
    batch_size = 128
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    dataset = NewClassTestDataset(data_dir)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 创建检测器并分析
    detector = UnknownDetector(model_path, threshold)
    results = detector.analyze_unknown(data_loader)
    
    # 打印分析结果
    print(f"\nUnknown Detection Analysis (threshold = {threshold})")
    print("-" * 50)
    print(f"Total samples: {results['total_samples']}")
    print("\nUnknown Detection by Class:")
    for label, unknown_count in results['detected_unknown_by_class'].items():
        total_count = results['total_samples_by_class'][label]
        unknown_rate = results['unknown_rate_by_class'][label]
        print(f"Class {label}: {unknown_count} / {total_count} unknown samples ({unknown_rate:.2f}%)")
    
    print(f"\nOverall Unknown Detection Rate: {results['overall_detection_rate']*100:.2f}%")
    
    # 保存结果
    with open(os.path.join(output_dir, 'unknown_detection_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {output_dir}")

if __name__ == '__main__':
    main()