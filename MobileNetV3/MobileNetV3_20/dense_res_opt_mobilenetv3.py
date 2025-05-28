import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HSwish(nn.Module):
    """H-Swish激活函数"""
    def forward(self, x):
        return x * F.relu6(x + 3) / 6

class HSigmoid(nn.Module):
    """H-Sigmoid激活函数"""
    def forward(self, x):
        return F.relu6(x + 3) / 6

class SEModule(nn.Module):
    """改进的SE注意力机制"""
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, 1, 0, bias=True),
            HSigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y.expand_as(x)

class MobileBottleneck(nn.Module):
    """改进的MobileNetV3瓶颈层"""
    def __init__(self, inp, oup, hidden_dim, kernel_size, stride, use_se, use_hs):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        activation = HSwish() if use_hs else nn.ReLU(inplace=True)

        # 点卷积升维
        self.conv1 = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            activation
        )

        # 密集连接
        self.dense_connect = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            activation
        )

        # 深度可分离卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size, stride, 
                     (kernel_size - 1) // 2, groups=hidden_dim * 2, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            SEModule(hidden_dim * 2) if use_se else nn.Identity(),
            activation
        )

        # 点卷积降维
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )

        # 添加dropout层增强正则化
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        y = self.conv1(x)
        z = self.dense_connect(x)
        y = torch.cat([y, z], dim=1)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.dropout(y)
        if self.identity:
            return x + y
        else:
            return y

class TrafficMobileNetV3(nn.Module):
    """针对网络流量优化的MobileNetV3"""
    def __init__(self, num_classes=11):
        super(TrafficMobileNetV3, self).__init__()
        
        # 初始层
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            HSwish()
        )

        # 构建网络配置
        # [inp, hidden_dim, oup, kernel, stride, use_se, use_hs]
        self.configs = [
            [16, 32, 16, 3, 1, True, False],   # 保持空间维度
            [16, 48, 24, 3, 1, False, False],  # 注意力逐步增加
            [24, 72, 24, 3, 1, True, True],    
            [24, 72, 40, 5, 2, True, True],    # 5x5感受野
            [40, 120, 40, 5, 1, True, True],   # 保持特征
            [40, 120, 80, 3, 1, False, True],  # 增加通道数
            [80, 240, 80, 3, 1, True, True],   # 深层特征
            [80, 200, 80, 3, 1, True, True],   # 保持维度
        ]

        # 构建网络层
        self.layers = []
        for config in self.configs:
            inp, hidden_dim, oup, kernel, stride, use_se, use_hs = config
            self.layers.append(
                MobileBottleneck(inp, oup, hidden_dim, kernel, stride, use_se, use_hs)
            )
        self.features = nn.Sequential(*self.layers)

        # 特征融合
        self.conv2 = nn.Sequential(
            nn.Conv2d(80, 160, 1, 1, 0, bias=False),
            nn.BatchNorm2d(160),
            HSwish()
        )

        # 自适应池化
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
            nn.Linear(64, num_classes)
        )

        # 权重初始化
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """改进的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

def test_model():
    """测试模型"""
    # 创建示例输入
    batch_size = 2
    x = torch.randn(batch_size, 1, 9, 9)
    
    # 初始化模型
    model = TrafficMobileNetV3(num_classes=11)
    model.eval()
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
    
    # 打印信息
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

if __name__ == "__main__":
    test_model()