from dense_res_opt_mobilenetv3 import TrafficMobileNetV3, HSwish, SEModule, MobileBottleneck
import torch
import torch.nn as nn
import torch.nn.functional as F

class IncrementalTrafficMobileNetV3(nn.Module):
    """支持增量学习的交通分类网络"""
    def __init__(self, old_model_path, old_num_classes=11, new_num_classes=12):
        super(IncrementalTrafficMobileNetV3, self).__init__()
        
        # 加载预训练模型
        self.old_model = TrafficMobileNetV3(num_classes=old_num_classes)
        checkpoint = torch.load(old_model_path, map_location='cpu')
        self.old_model.load_state_dict(checkpoint['model_state_dict'])
        
        # 冻结旧模型参数
        for param in self.old_model.parameters():
            param.requires_grad = False
            
        # 复制基础特征提取器
        self.conv1 = self.old_model.conv1
        self.features = self.old_model.features
        
        # 新的特征融合层(可训练)
        self.conv2 = nn.Sequential(
            nn.Conv2d(80, 160, 1, 1, 0, bias=False),
            nn.BatchNorm2d(160),
            HSwish()
        )
        
        # 复制预训练权重到conv2
        self.conv2[0].weight.data = self.old_model.conv2[0].weight.data.clone()
        self.conv2[1].weight.data = self.old_model.conv2[1].weight.data.clone()
        self.conv2[1].bias.data = self.old_model.conv2[1].bias.data.clone()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 新的分类器
        self.classifier = nn.Sequential(
            nn.Linear(160, 128),
            nn.BatchNorm1d(128),
            HSwish(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            HSwish(),
            nn.Dropout(0.2),
            nn.Linear(64, new_num_classes)
        )
        
        # 初始化新分类器的权重
        self._initialize_new_classifier()
        
        # 正确复制旧模型的权重到对应层
        with torch.no_grad():
            # 复制第一个线性层和BatchNorm层的权重
            self.classifier[0].weight.data = self.old_model.classifier[0].weight.data.clone()
            self.classifier[0].bias.data = self.old_model.classifier[0].bias.data.clone()
            self.classifier[1].weight.data = self.old_model.classifier[1].weight.data.clone()
            self.classifier[1].bias.data = self.old_model.classifier[1].bias.data.clone()
            
            # 复制第二个线性层和BatchNorm层的权重
            self.classifier[4].weight.data = self.old_model.classifier[4].weight.data.clone()
            self.classifier[4].bias.data = self.old_model.classifier[4].bias.data.clone()
            self.classifier[5].weight.data = self.old_model.classifier[5].weight.data.clone()
            self.classifier[5].bias.data = self.old_model.classifier[5].bias.data.clone()
            
            # 复制最后一个线性层的权重（仅复制旧类别的部分）
            self.classifier[-1].weight.data[:old_num_classes, :] = self.old_model.classifier[-1].weight.data
            self.classifier[-1].bias.data[:old_num_classes] = self.old_model.classifier[-1].bias.data
        
        self.old_num_classes = old_num_classes
        self.new_num_classes = new_num_classes
        
    def _initialize_new_classifier(self):
        """初始化新分类器权重"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
    def get_old_model_output(self, x):
        """获取旧模型的输出"""
        with torch.no_grad():
            return self.old_model(x)
            
    def get_features(self, x):
        """提取特征"""
        x = self.conv1(x)
        x = self.features(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)
        
    def forward(self, x):
        x = self.get_features(x)
        x = self.classifier(x)
        return x
        
    def get_old_new_outputs(self, x):
        """同时获取新旧模型的输出"""
        new_output = self.forward(x)
        old_output = self.get_old_model_output(x)
        return old_output, new_output