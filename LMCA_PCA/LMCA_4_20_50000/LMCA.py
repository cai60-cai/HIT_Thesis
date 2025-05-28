import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpandFusionConv(nn.Module):
    """扩展融合卷积层(Conv2)
    将少通道特征转换为多通道特征并进行融合"""
    def __init__(self, in_channels, expand_ratio):
        super().__init__()
        self.mid_channels = int(in_channels * expand_ratio)  # 按公式(3)计算中间通道数
        
        # 第一个并行分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU6(inplace=True)
        )
        
        # 第二个并行分支
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU6(inplace=True)
        )
        
    def forward(self, x):
        # 并行分支处理
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        
        # 特征融合（元素级乘法）
        out = x1 * x2
        return out

class EnhancedSeparableConv3(nn.Module):
    """改进的可分离卷积层，确保特征图大小适当"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        self.stride = stride
        self.mid_channels = in_channels // 2
        
        # pool1和pool2分支的卷积
        self.pool1_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels, kernel_size=1),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU6(inplace=True)
        )
        
        self.pool2_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels, kernel_size=1),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU6(inplace=True)
        )
        
        # 主干部分的卷积层
        self.conv_main = nn.Sequential(
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=1),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU6(inplace=True)
        )
        
        # 分离后的conv+expand模块
        self.conv_expand1 = nn.Sequential(
            nn.Conv2d(self.mid_channels // 2, out_channels // 2, kernel_size=3, 
                     stride=stride, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU6(inplace=True)
        )
        
        self.conv_expand2 = nn.Sequential(
            nn.Conv2d(self.mid_channels // 2, out_channels // 2, kernel_size=3, 
                     stride=stride, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU6(inplace=True)
        )
        
        # 最终融合的变换
        self.final_fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 调整池化大小，确保不会太小
        pool1_size = max(H//2, 2)
        pool2_size = max(H//2, 2)  # 改为和pool1相同大小
        
        # 池化分支
        pool1 = F.adaptive_avg_pool2d(x, output_size=(pool1_size, pool1_size))
        pool1 = self.pool1_conv(pool1)
        
        pool2 = F.adaptive_avg_pool2d(x, output_size=(pool2_size, pool2_size))
        pool2 = self.pool2_conv(pool2)
        
        # 确保大小匹配
        if pool1.size(2) != pool2.size(2):
            pool2 = F.interpolate(pool2, size=(pool1.size(2), pool1.size(3)), 
                                mode='bilinear', align_corners=False)
        
        # 池化分支融合
        pooled = pool1 * pool2
        
        # 上采样回原始大小
        if self.stride == 1:
            target_size = (H, W)
        else:
            target_size = (H//self.stride, W//self.stride)
            
        pooled = F.interpolate(pooled, size=target_size, 
                             mode='bilinear', align_corners=False)
        
        # 主干处理
        main = self.conv_main(pooled)
        
        # 特征分离和扩展
        split1, split2 = torch.chunk(main, 2, dim=1)
        
        # 分别处理两个分支
        expand1 = self.conv_expand1(split1)
        expand2 = self.conv_expand2(split2)
        
        # 特征融合
        fused = torch.cat([expand1, expand2], dim=1)
        
        # 最终变换
        out = self.final_fusion(fused)
        
        return out



class RecoveryConv4(nn.Module):
    """恢复卷积层(Conv4)
    实现通道数降低和特征提取，不使用ReLU"""
    def __init__(self, in_channels, recovery_ratio):
        super().__init__()
        # 根据公式(13)计算输出通道数
        self.out_channels = int(in_channels * recovery_ratio)
        
        # 1x1卷积进行通道恢复
        self.conv = nn.Conv2d(
            in_channels, 
            self.out_channels, 
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False  # 因为后面会用BatchNorm
        )
        
        # 使用BatchNorm进行归一化，按照公式(14)
        self.bn = nn.BatchNorm2d(self.out_channels)
        
    def forward(self, x):
        # 通道恢复
        x = self.conv(x)
        # 归一化操作
        x = self.bn(x)
        # 注意：这里不使用ReLU
        return x

class ImprovedCoordinateAttention(nn.Module):
    """改进的坐标注意力机制"""
    def __init__(self, in_channels, reduction_ratio=32):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        
        # 计算中间通道数
        self.inter_channels = max(8, in_channels // reduction_ratio)
        
        # F1转换函数 (公式9)
        self.F1 = nn.Sequential(
            nn.Conv2d(in_channels, self.inter_channels, 1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU6(inplace=True)
        )
        
        # Fh和Fw转换函数 (公式10,11)
        self.Fh = nn.Conv2d(self.inter_channels, in_channels, 1)
        self.Fw = nn.Conv2d(self.inter_channels, in_channels, 1)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # 计算水平方向池化 (公式7)
        z_h = x.mean(dim=3, keepdim=True)  # [B,C,H,1]
        
        # 计算垂直方向池化 (公式8)
        z_w = x.mean(dim=2, keepdim=True)  # [B,C,1,W]
        
        # 拼接并通过F1 (公式9)
        z = torch.cat([z_h.squeeze(3), z_w.squeeze(2)], dim=2)  # [B,C,H+W]
        z = z.unsqueeze(3)  # [B,C,H+W,1]
        z = self.F1(z)
        
        # 分离特征 (公式10,11)
        z_h, z_w = torch.split(z, [height, width], dim=2)
        z_h = z_h.squeeze(3)
        z_w = z_w.squeeze(2)
        
        # 生成注意力权重
        g_h = self.sigmoid(self.Fh(z_h.unsqueeze(3)))  # [B,C,H,1]
        g_w = self.sigmoid(self.Fw(z_w.unsqueeze(2)))  # [B,C,1,W]
        
        # 应用注意力 (公式12)
        out = x * g_h * g_w
        
        return out

class CirculationBlock(nn.Module):
    """改进的循环块，处理特征图大小不匹配问题"""
    def __init__(self, in_channels, exp_ratio, out_channels, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = int(in_channels * exp_ratio)
        self.out_channels = out_channels
        self.stride = stride
        
        # Conv2: 扩展融合层
        self.expand_fusion = ExpandFusionConv(in_channels, exp_ratio)
        
        # Conv3: 使用增强的可分离卷积
        self.conv3 = EnhancedSeparableConv3(
            self.mid_channels, 
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1
        )
        
        # 跳跃连接
        self.use_shortcut = stride == 1 and in_channels == out_channels
        
        # 下采样层（如果需要）
        self.downsample = None
        if not self.use_shortcut:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=3,  # 使用3x3卷积而不是1x1
                    stride=stride,
                    padding=1
                ),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        # 主路径
        out = self.expand_fusion(x)
        out = self.conv3(out)
        
        # 残差连接
        if self.use_shortcut:
            out = out + identity
        elif self.downsample is not None:
            # 确保identity经过下采样后与out大小匹配
            identity = self.downsample(identity)
            out = out + identity
        
        return out

class SimulatedConv1(nn.Module):
    """初始卷积层(Conv1)
    将单通道数据转换为多通道特征，模拟数据为图像形式
    """
    def __init__(self, out_channels=32):
        super().__init__()
        # 按照论文描述，输入为单通道
        self.in_channels = 1
        self.out_channels = out_channels
        
        # 卷积层实现
        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,  # 使用3x3卷积核
            stride=2,       # 步长为2，降低空间维度
            padding=1,      # 使用padding保持特征
            bias=False      # 使用BatchNorm时不需要bias
        )
        
        # 批归一化层
        self.bn = nn.BatchNorm2d(self.out_channels)
        
        # 激活函数
        self.relu = nn.ReLU6(inplace=True)
        
    def forward(self, x):
        # 卷积操作
        x = self.conv(x)
        # 归一化
        x = self.bn(x)
        # 激活
        x = self.relu(x)
        return x


class LMCA(nn.Module):
    """改进的LMCA模型"""
    def __init__(self, num_classes=13):
        super().__init__()
        
        # 初始卷积 (Conv1)
        self.conv1 = SimulatedConv1(out_channels=32)
        
        # 循环层配置 - 调整步长策略
        # 修改LMCA类中的循环层配置
        self.circulation_configs = [
            # (in_ch, exp_ratio, out_ch, stride)
            (32, 6, 24, 2),  # s=2 (表2中第一行)
            (24, 6, 48, 2),  # s=2 (表2中第二行)
            (48, 6, 72, 2),  # s=2 (表2中第三行)
            (72, 6, 96, 1)   # s=1 (表2中第四行)
        ]
        
        # 循环层
        self.circulation_layers = self._make_circulation_layers()
        
        # 坐标注意力
        self.coord_attention = ImprovedCoordinateAttention(96)
        
        # Conv4 恢复卷积层
        self.conv4 = RecoveryConv4(96, recovery_ratio=0.75)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(72, num_classes)
        )
    
    def _make_circulation_layers(self):
        layers = []
        for in_ch, exp_ratio, out_ch, stride in self.circulation_configs:
            layers.append(CirculationBlock(in_ch, exp_ratio, out_ch, stride))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # print(f"Input: {x.shape}")
        
        x = self.conv1(x)
        # print(f"After Conv1: {x.shape}")
        
        for i, layer in enumerate(self.circulation_layers):
            x = layer(x)
            # print(f"After Circulation {i+1}: {x.shape}")
        
        x = self.coord_attention(x)
        # print(f"After Attention: {x.shape}")
        
        x = self.conv4(x)
        # print(f"After Conv4: {x.shape}")
        
        x = self.global_pool(x)
        # print(f"After Pool: {x.shape}")
        
        x = self.classifier(x)
        # print(f"Final output: {x.shape}")
        
        return x

# def test_model():
#     """测试模型"""
#     # 创建示例输入
#     batch_size = 2
#     input_channels = 1
#     input_size = 9
#     x = torch.randn(batch_size, input_channels, input_size, input_size)
    
#     print(f"\nInput shape: {x.shape}")
    
#     # 初始化模型
#     model = LMCA(num_classes=13)
#     model.eval()  # 设置为评估模式
    
#     # 追踪每一层的输出大小
#     with torch.no_grad():
#         output = model(x)
    
#     print(f"Output shape: {output.shape}")
    
#     # 计算参数量
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"\nTotal parameters: {total_params:,}")

# if __name__ == "__main__":
#     test_model()

# Input shape: torch.Size([2, 1, 9, 9])
# Input: torch.Size([2, 1, 9, 9])
# After Conv1: torch.Size([2, 32, 5, 5])
# After Circulation 1: torch.Size([2, 24, 3, 3])
# After Circulation 2: torch.Size([2, 48, 2, 2])
# After Circulation 3: torch.Size([2, 72, 1, 1])
# After Circulation 4: torch.Size([2, 96, 1, 1])
# After Attention: torch.Size([2, 96, 1, 1])
# After Conv4: torch.Size([2, 72, 1, 1])
# After Pool: torch.Size([2, 72, 1, 1])
# Final output: torch.Size([2, 13])
# Output shape: torch.Size([2, 13])

# Total parameters: 835,901