import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from LMCA import LMCA  # 导入模型
import json

class NetworkTrafficDataset(Dataset):
    """网络流量数据集加载器"""
    def __init__(self, data_dir, prefix):
        self.data_dir = data_dir
        self.prefix = prefix
        
        # 加载所有批次的特征
        self.features = []
        batch_idx = 0
        while True:
            batch_path = f"{data_dir}/{prefix}_features_batch_{batch_idx}.npy"
            if not os.path.exists(batch_path):
                break
            self.features.append(np.load(batch_path))
            batch_idx += 1
        
        # 合并所有批次
        self.features = np.concatenate(self.features, axis=0)
        
        # 加载标签
        self.labels = np.load(f"{data_dir}/{prefix}_labels.npy")
        
        # 确保数据和标签长度匹配
        assert len(self.features) == len(self.labels), "特征和标签数量不匹配"
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx]).unsqueeze(0)  # 添加通道维度
        label = torch.LongTensor([self.labels[idx]])[0]
        return feature, label


def evaluate(model, dataloader, criterion, device, num_classes):
    """评估模型性能"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc='Testing'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            
            preds = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(target.cpu().numpy())
    
    # 计算指标
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    
    return total_loss / len(dataloader), precision, recall, accuracy, f1, cm


def plot_confusion_matrix(cm, class_names, save_path):
    """绘制并保存混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def test_multiple_epochs(data_dir, model_dir, output_dir, num_epochs, batch_size, num_classes, class_names):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载测试数据集
    print("Loading test dataset...")
    test_dataset = NetworkTrafficDataset(data_dir, 'test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        epoch_dir = f"{model_dir}/epoch_{epoch}"
        model_path = f"{epoch_dir}/model.pth"
        output_epoch_dir = f"{output_dir}/epoch_{epoch}"
        
        # 确保输出目录存在
        os.makedirs(output_epoch_dir, exist_ok=True)
        
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            continue

        print(f"Evaluating model from {model_path}...")
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=device)
        model = LMCA(num_classes=num_classes).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 评估模型性能
        test_loss, precision, recall, accuracy, f1, cm = evaluate(model, test_loader, criterion, device, num_classes)
        
        # 保存测试结果
        results = {
            'test_loss': test_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        results_path = f"{output_epoch_dir}/test_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Test results saved to {results_path}")
        
        # 绘制并保存混淆矩阵
        cm_path = f"{output_epoch_dir}/confusion_matrix.png"
        plot_confusion_matrix(cm, class_names, cm_path)


def main():
    # 配置参数
    data_dir = 'processed_data'
    model_dir = 'models_avg'
    output_dir = 'evaluation_results_avg'
    num_epochs = 20
    batch_size = 256
    num_classes = 11  # 替换为你模型的实际类别数
    class_names = [f"Class {i}" for i in range(num_classes)]  # 替换为实际类别名
    
    test_multiple_epochs(data_dir, model_dir, output_dir, num_epochs, batch_size, num_classes, class_names)


if __name__ == '__main__':
    main()
