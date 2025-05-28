import torch
import torch.nn as nn
import torch.nn.functional as F

class HSwish(nn.Module):
    """H-Swish激活函数"""
    def forward(self, x):
        return x * F.relu6(x + 3) / 6

class HSigmoid(nn.Module):
    """H-Sigmoid激活函数"""
    def forward(self, x):
        return F.relu6(x + 3) / 6

class ChannelAttention(nn.Module):
    """通道注意力机制"""
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    """空间注意力机制"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return self.sigmoid(attention)

class CBAM(nn.Module):
    """CBAM注意力机制"""
    def __init__(self, channel, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out

class SpatialGate(nn.Module):
    """空间门控模块"""
    def __init__(self, channel, reduction=16):
        super(SpatialGate, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel // reduction, 1, bias=False)
        self.conv2 = nn.Conv2d(channel // reduction, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return self.sigmoid(out)

class ChannelGate(nn.Module):
    """通道门控模块"""
    def __init__(self, channel, reduction=16):
        super(ChannelGate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class ASCNet(nn.Module):
    """A-SCNet注意力机制"""
    def __init__(self, channel, reduction=16):
        super(ASCNet, self).__init__()
        self.spatial_gate = SpatialGate(channel, reduction)
        self.channel_gate = ChannelGate(channel, reduction)

    def forward(self, x):
        spatial_attention = self.spatial_gate(x)
        channel_attention = self.channel_gate(x)
        return x * spatial_attention * channel_attention

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
            # 引入CBAM注意力机制
            CBAM(hidden_dim) if use_se else nn.Identity(),
            # 引入A-SCNet注意力机制
            ASCNet(hidden_dim) if use_hs else nn.Identity(),
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

        # MobileNetV3-Small配置
        self.mobile_bottlenecks = nn.Sequential(
            MobileBottleneck(16, 16, 32, 3, 2, True, False),   # 1
            MobileBottleneck(16, 24, 48, 3, 2, False, False),  # 2
            MobileBottleneck(24, 24, 72, 3, 1, True, True),    # 3
            MobileBottleneck(24, 40, 72, 5, 2, True, True),    # 4
            # 在此处引入A-SCNet注意力机制
            MobileBottleneck(40, 40, 120, 5, 1, True, True),   # 5
            MobileBottleneck(40, 40, 120, 5, 1, True, True),   # 6
            MobileBottleneck(40, 48, 120, 5, 1, True, True),   # 7
            MobileBottleneck(48, 48, 144, 5, 1, True, True),   # 8
            MobileBottleneck(48, 96, 288, 5, 2, True, True),   # 9
            MobileBottleneck(96, 96, 576, 5, 1, True, True),   # 10
            MobileBottleneck(96, 96, 576, 5, 1, True, True),   # 11
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(576),
            HSwish()
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 引入Transformer模块
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=576, nhead=4), num_layers=2)

        # 分类器
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(576),
            nn.Linear(576, 128),
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
        x = self.mobile_bottlenecks(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # 引入Transformer模块
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)

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