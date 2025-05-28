
import torch
import torch.nn as nn
import torch.nn.functional as F

class HSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3) / 6

class HSigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3) / 6

class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            HSigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        if nl == 'RE':
            self.nlin_layer = nn.ReLU(inplace=True)
        elif nl == 'HS':
            self.nlin_layer = HSwish()
        else:
            self.nlin_layer = nn.ReLU(inplace=True)

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, exp, 1, 1, 0, bias=False),
            nn.BatchNorm2d(exp),
            self.nlin_layer,
            # dw
            nn.Conv2d(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            nn.BatchNorm2d(exp),
            SEModule(exp) if se else nn.Sequential(),
            self.nlin_layer,
            # pw-linear
            nn.Conv2d(exp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV3(nn.Module):
    def __init__(self, num_classes=11):
        super(MobileNetV3, self).__init__()
        # 修改第一层以适应单通道输入
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            HSwish()
        )

        # MobileNetV3-Small配置
        self.mobile_bottlenecks = nn.Sequential(
            MobileBottleneck(16, 16, 3, 2, 16, False, 'RE'),    # 1
            MobileBottleneck(16, 24, 3, 2, 72, False, 'RE'),    # 2
            MobileBottleneck(24, 24, 3, 1, 88, False, 'RE'),    # 3
            MobileBottleneck(24, 40, 5, 2, 96, True, 'HS'),     # 4
            MobileBottleneck(40, 40, 5, 1, 240, True, 'HS'),    # 5
            MobileBottleneck(40, 40, 5, 1, 240, True, 'HS'),    # 6
            MobileBottleneck(40, 48, 5, 1, 120, True, 'HS'),    # 7
            MobileBottleneck(48, 48, 5, 1, 144, True, 'HS'),    # 8
            MobileBottleneck(48, 96, 5, 2, 288, True, 'HS'),    # 9
            MobileBottleneck(96, 96, 5, 1, 576, True, 'HS'),    # 10
            MobileBottleneck(96, 96, 5, 1, 576, True, 'HS'),    # 11
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(576),
            HSwish()
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            HSwish(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.mobile_bottlenecks(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
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
    model = MobileNetV3(num_classes=11)
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
