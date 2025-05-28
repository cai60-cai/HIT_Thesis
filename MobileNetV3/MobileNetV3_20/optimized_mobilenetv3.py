
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

        # 深度可分离卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                     (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            SEModule(hidden_dim) if use_se else nn.Identity(),
            activation
        )

        # 点卷积降维
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )

        # 添加dropout层增强正则化
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        y = self.conv1(x)
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

# def test_model():
#     """测试模型"""
#     model = TrafficMobileNetV3()
#     x = torch.randn(2, 1, 9, 9)
#     y = model(x)
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {y.shape}")
#     # 计算参数量
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"Total parameters: {total_params:,}")
    
# if __name__ == "__main__":
#     test_model()


# 1. 初始卷积层优化：
#    - 第一个版本的 `TrafficMobileNetV3` 将输入通道数从 3 通道降低到了 1 通道,以更好地适应单通道的网络流量数据。

# 2. 瓶颈层优化：
#    - 第一个版本的 `MobileBottleneck` 层进行了更细致的设计,包括:
#      - 调整了输入、隐藏层和输出通道数
#      - 针对不同的卷积核大小和步长进行了配置
#      - 增加了是否使用SE模块和HS激活的选项

# 3. 激活函数优化：
#    - 两个版本都使用了改进的 `HSwish` 和 `HSigmoid` 激活函数,相比标准的 ReLU 和 Sigmoid 具有更平滑的非线性特性。

# 4. 注意力机制优化：
#    - 第一个版本在 `MobileBottleneck` 中使用了改进的 `SEModule` 注意力机制,增强了模型对重要特征的关注。

# 5. 正则化优化：
#    - 第一个版本在 `MobileBottleneck` 中增加了 Dropout 层,进一步提高了模型的泛化能力。

# 6. 权重初始化优化：
#    - 两个版本都使用了改进的权重初始化方法,有利于模型收敛更快并获得更好的性能。
