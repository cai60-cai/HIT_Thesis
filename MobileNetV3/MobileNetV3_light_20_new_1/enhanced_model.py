# from dense_res_opt_mobilenetv3 import TrafficMobileNetV3, HSwish, SEModule, MobileBottleneck
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class WebAttackFeatureExtractor(nn.Module):
#     """专门用于Web攻击特征提取的模块"""
#     def __init__(self, in_channels):
#         super(WebAttackFeatureExtractor, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, 32, 3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#         )
        
#     def forward(self, x):
#         return self.conv(x)

# class IncrementalTrafficMobileNetV3(nn.Module):
#     """支持增量学习的交通分类网络"""
#     def __init__(self, old_model_path, old_num_classes=11, new_num_classes=12):
#         super(IncrementalTrafficMobileNetV3, self).__init__()
        
#         # 加载预训练模型
#         self.old_model = TrafficMobileNetV3(num_classes=old_num_classes)
#         checkpoint = torch.load(old_model_path, map_location='cpu')
#         self.old_model.load_state_dict(checkpoint['model_state_dict'])
        
#         # 冻结旧模型参数
#         for param in self.old_model.parameters():
#             param.requires_grad = False
            
#         # 复制基础特征提取器
#         self.conv1 = self.old_model.conv1
#         self.features = self.old_model.features
        
#         # 添加Web攻击特征提取器
#         self.web_extractor = WebAttackFeatureExtractor(1)
        
#         # 新的特征融合层(整合Web攻击特征)
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(144, 160, 1, 1, 0, bias=False),  # 80(原始) + 64(web) = 144
#             nn.BatchNorm2d(160),
#             HSwish()
#         )
        
#         # 初始化conv2的部分权重
#         with torch.no_grad():
#             # 对原有80通道的权重进行复制
#             self.conv2[0].weight.data[:, :80, :, :] = self.old_model.conv2[0].weight.data.clone()
#             # 对新增的64通道随机初始化（已通过默认初始化完成）
#             self.conv2[1].weight.data = self.old_model.conv2[1].weight.data.clone()
#             self.conv2[1].bias.data = self.old_model.conv2[1].bias.data.clone()
        
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
#         # 新的分类器
#         self.classifier = nn.Sequential(
#             nn.Linear(160, 128),
#             nn.BatchNorm1d(128),
#             HSwish(),
#             nn.Dropout(0.2),
#             nn.Linear(128, 64),
#             nn.BatchNorm1d(64),
#             HSwish(),
#             nn.Dropout(0.2),
#             nn.Linear(64, new_num_classes)
#         )
        
#         # 初始化新分类器的权重
#         self._initialize_new_classifier()
        
#         # 正确复制旧模型的权重到对应层
#         with torch.no_grad():
#             # 复制第一个线性层和BatchNorm层的权重
#             self.classifier[0].weight.data = self.old_model.classifier[0].weight.data.clone()
#             self.classifier[0].bias.data = self.old_model.classifier[0].bias.data.clone()
#             self.classifier[1].weight.data = self.old_model.classifier[1].weight.data.clone()
#             self.classifier[1].bias.data = self.old_model.classifier[1].bias.data.clone()
            
#             # 复制第二个线性层和BatchNorm层的权重
#             self.classifier[4].weight.data = self.old_model.classifier[4].weight.data.clone()
#             self.classifier[4].bias.data = self.old_model.classifier[4].bias.data.clone()
#             self.classifier[5].weight.data = self.old_model.classifier[5].weight.data.clone()
#             self.classifier[5].bias.data = self.old_model.classifier[5].bias.data.clone()
            
#             # 复制最后一个线性层的权重（仅复制旧类别的部分）
#             self.classifier[-1].weight.data[:old_num_classes, :] = self.old_model.classifier[-1].weight.data
#             self.classifier[-1].bias.data[:old_num_classes] = self.old_model.classifier[-1].bias.data
        
#         self.old_num_classes = old_num_classes
#         self.new_num_classes = new_num_classes
        
#     def _initialize_new_classifier(self):
#         """初始化新分类器权重"""
#         for m in self.classifier.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)
                
#     def get_old_model_output(self, x):
#         """获取旧模型的输出"""
#         with torch.no_grad():
#             return self.old_model(x)
            
#     def get_features(self, x):
#         """提取特征"""
#         # 基础特征提取
#         x_base = self.conv1(x)
#         x_base = self.features(x_base)
        
#         # Web攻击特征提取
#         x_web = self.web_extractor(x)
        
#         # 确保特征图大小一致
#         if x_web.size(-1) != x_base.size(-1):
#             x_web = F.adaptive_avg_pool2d(x_web, (x_base.size(-2), x_base.size(-1)))
        
#         # 特征融合
#         x = torch.cat([x_base, x_web], dim=1)
#         x = self.conv2(x)
#         x = self.avgpool(x)
#         return x.view(x.size(0), -1)
        
#     def forward(self, x):
#         x = self.get_features(x)
#         x = self.classifier(x)
#         return x
        
#     def get_old_new_outputs(self, x):
#         """同时获取新旧模型的输出"""
#         new_output = self.forward(x)
#         old_output = self.get_old_model_output(x)
#         return old_output, new_output


##########################13
from dense_res_opt_mobilenetv3 import TrafficMobileNetV3, HSwish, SEModule, MobileBottleneck
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

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

class IncrementalTrafficMobileNetV3(nn.Module):
    def __init__(self, old_model_path, old_num_classes=11, new_num_classes=13):
        super(IncrementalTrafficMobileNetV3, self).__init__()
        
        # 加载预训练权重
        checkpoint = torch.load(old_model_path, map_location='cpu')
        old_state_dict = checkpoint['model_state_dict']
        
        # 创建基础网络组件
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            HSwish()
        )
        
        # 特征提取层
        self.features = TrafficMobileNetV3(num_classes=old_num_classes).features
        
        # Web攻击特征提取器
        self.web_extractor = WebAttackFeatureExtractor(1)
        
        # 特征融合层
        self.conv2 = nn.Sequential(
            nn.Conv2d(144, 160, 1, 1, 0, bias=False),  # 80(原始) + 64(web) = 144
            nn.BatchNorm2d(160),
            HSwish()
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
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
        
        # 创建用于知识蒸馏的旧模型 (使用原始的类别数)
        self.old_model = TrafficMobileNetV3(num_classes=old_num_classes)
        
        # 处理旧模型的状态字典
        processed_state_dict = {}
        for k, v in old_state_dict.items():
            if k.startswith('old_model.'):
                new_key = k[10:]  # 移除 'old_model.' 前缀
                if 'classifier.8' in new_key:  # 跳过最后一层分类器
                    continue
                processed_state_dict[new_key] = v
        
        # 加载处理后的状态字典到旧模型
        missing_keys, unexpected_keys = self.old_model.load_state_dict(processed_state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        
        # 冻结旧模型
        for param in self.old_model.parameters():
            param.requires_grad = False
            
        # 加载主模型的特征提取部分
        self._load_feature_extractor(old_state_dict)
        
        self.old_num_classes = old_num_classes
        self.new_num_classes = new_num_classes
    
    def _load_feature_extractor(self, state_dict):
        """加载特征提取器的权重"""
        # 处理conv1层
        for name, param in self.conv1.named_parameters():
            old_name = f'old_model.conv1.{name}'
            if old_name in state_dict:
                param.data.copy_(state_dict[old_name])
        
        # 处理features层
        for name, param in self.features.named_parameters():
            old_name = f'old_model.features.{name}'
            if old_name in state_dict:
                param.data.copy_(state_dict[old_name])
        
        # 处理BatchNorm统计量
        for name, buf in self.features.named_buffers():
            old_name = f'old_model.features.{name}'
            if old_name in state_dict:
                buf.data.copy_(state_dict[old_name])
    
    def forward(self, x):
        # 基础特征提取
        x_base = self.conv1(x)
        x_base = self.features(x_base)
        
        # Web攻击特征提取
        x_web = self.web_extractor(x)
        
        # 确保特征图大小一致
        if x_web.size(-1) != x_base.size(-1):
            x_web = F.adaptive_avg_pool2d(x_web, (x_base.size(-2), x_base.size(-1)))
        
        # 特征融合
        x = torch.cat([x_base, x_web], dim=1)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def get_old_new_outputs(self, x):
        """同时获取新旧模型的输出"""
        new_output = self.forward(x)
        with torch.no_grad():
            old_output = self.old_model(x)
        return old_output, new_output