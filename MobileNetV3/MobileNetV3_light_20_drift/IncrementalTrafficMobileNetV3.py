
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from dense_res_opt_mobilenetv3 import HSwish, TrafficMobileNetV3

class FeatureAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.LayerNorm(in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        weights = self.attention(x)
        return x * weights

class FeatureEnhancement(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, in_channels)
        self.ln1 = nn.LayerNorm(in_channels)
        self.fc2 = nn.Linear(in_channels, in_channels)
        self.ln2 = nn.LayerNorm(in_channels)
        
    def forward(self, x):
        identity = x
        x = F.gelu(self.ln1(self.fc1(x)))
        x = self.ln2(self.fc2(x))
        return x + identity

class ContrastiveLearningHead(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, out_channels)
        )
        
    def forward(self, x):
        return F.normalize(self.projection(x), dim=1)

class IncrementalTrafficMobileNetV3(nn.Module):
    def __init__(self, base_model, num_old_classes=11, num_new_classes=2):
        super().__init__()
        
        self.num_classes = num_old_classes + num_new_classes
        self.num_old_classes = num_old_classes
        self.num_new_classes = num_new_classes
        self.device = next(base_model.parameters()).device
        
        # 基础特征提取器
        self.conv1 = copy.deepcopy(base_model.conv1)
        self.features = copy.deepcopy(base_model.features)
        self.conv2 = copy.deepcopy(base_model.conv2)
        self.avgpool = copy.deepcopy(base_model.avgpool)
        
        # 特征处理
        self.feature_attention = FeatureAttention(160)
        self.feature_enhancement = FeatureEnhancement(160)
        
        # 对比学习头部
        self.contrastive_head = ContrastiveLearningHead(160)
        
        # 分类器 - 使用原始维度
        self.classifier = nn.Sequential(
            nn.Linear(160, 128),
            nn.LayerNorm(128),
            HSwish(),
            FeatureAttention(128),
            FeatureEnhancement(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            HSwish(),
            FeatureAttention(64),
            FeatureEnhancement(64),
            nn.Dropout(0.3),
            nn.Linear(64, self.num_classes)
        )
        
        self._initialize_new_classifier(base_model)
        self.params_old = {}
        self.fisher = None
        
        # 移动到正确的设备
        self.to(self.device)
        
    def _initialize_new_classifier(self, base_model):
        # 第一层: 160 -> 128
        self.classifier[0].weight.data[:128, :160].copy_(
            base_model.classifier[0].weight.data
        )
        self.classifier[0].bias.data[:128].copy_(
            base_model.classifier[0].bias.data
        )
        
        # 中间层: 128 -> 64
        self.classifier[6].weight.data[:64, :128].copy_(
            base_model.classifier[4].weight.data
        )
        self.classifier[6].bias.data[:64].copy_(
            base_model.classifier[4].bias.data
        )
        
        # 最后一层: 处理新类别
        old_weight = base_model.classifier[-1].weight.data  # [11, 64]
        old_bias = base_model.classifier[-1].bias.data
        
        # 初始化新类别权重
        new_weight = torch.zeros(self.num_new_classes, 64).normal_(
            0, 0.01
        ).to(self.device)
        new_bias = torch.zeros(self.num_new_classes).to(self.device)
        
        self.classifier[-1].weight.data[:self.num_old_classes].copy_(old_weight)
        self.classifier[-1].weight.data[self.num_old_classes:].copy_(new_weight)
        self.classifier[-1].bias.data[:self.num_old_classes].copy_(old_bias)
        self.classifier[-1].bias.data[self.num_old_classes:].copy_(new_bias)

    def get_features(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.feature_attention(x)
        x = self.feature_enhancement(x)
        return x

    def forward(self, x, return_features=False):
        features = self.get_features(x)
        outputs = self.classifier(features)
        
        if return_features:
            return outputs, features
        return outputs

    def freeze_old_params(self):
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.conv2.parameters():
            param.requires_grad = False
            
        # 保持特征处理模块可训练
        for param in self.feature_attention.parameters():
            param.requires_grad = True
        for param in self.feature_enhancement.parameters():
            param.requires_grad = True
        
        # 分类器最后一层可训练
        for param in self.classifier[-1].parameters():
            param.requires_grad = True

    def partial_unfreeze(self, num_layers=2):
        self.freeze_old_params()
        trainable_layers = list(self.classifier.children())[-num_layers:]
        for layer in trainable_layers:
            for param in layer.parameters():
                param.requires_grad = True

    def unfreeze_all_params(self):
        for param in self.parameters():
            param.requires_grad = True

    def set_fisher_params(self, fisher, params_old):
        self.fisher = fisher
        self.params_old = params_old

def create_incremental_model(base_model_path, device, num_old_classes=11, num_new_classes=2):
    base_model = TrafficMobileNetV3(num_classes=num_old_classes)
    checkpoint = torch.load(base_model_path, map_location=device)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model = base_model.to(device)
    
    return IncrementalTrafficMobileNetV3(
        base_model, 
        num_old_classes=num_old_classes,
        num_new_classes=num_new_classes
    ).to(device)
