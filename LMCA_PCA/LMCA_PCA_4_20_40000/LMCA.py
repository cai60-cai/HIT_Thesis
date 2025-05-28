import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        self.stride = stride
        self.min_size = 2  # 设置最小尺寸
        
        # 调整中间通道数
        self.mid_channels = max(in_channels // 2, 8)  # 确保至少有8个通道
        
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
        
        # 分离后的conv+expand模块，使用1x1卷积替代3x3卷积
        half_channels = max(self.mid_channels // 2, 4)
        half_out_channels = max(out_channels // 2, 4)
        
        self.conv_expand1 = nn.Sequential(
            nn.Conv2d(half_channels, half_out_channels, kernel_size=1),
            nn.BatchNorm2d(half_out_channels),
            nn.ReLU6(inplace=True)
        )
        
        self.conv_expand2 = nn.Sequential(
            nn.Conv2d(half_channels, half_out_channels, kernel_size=1),
            nn.BatchNorm2d(half_out_channels),
            nn.ReLU6(inplace=True)
        )
        
        # 最终融合的变换
        self.final_fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 确保最小尺寸
        pool1_size = max(H//2, self.min_size)
        pool2_size = max(H//2, self.min_size)
        
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
        
        # 计算目标大小，确保至少为2x2
        if self.stride == 1:
            target_size = (max(H, self.min_size), max(W, self.min_size))
        else:
            target_size = (max(H//self.stride, self.min_size), 
                          max(W//self.stride, self.min_size))
            
        # 上采样回目标大小
        if pooled.size(2) != target_size[0] or pooled.size(3) != target_size[1]:
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
        
        # F1转换函数
        self.F1 = nn.Sequential(
            nn.Conv2d(in_channels, self.inter_channels, 1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU6(inplace=True)
        )
        
        # Fh和Fw转换函数
        self.Fh = nn.Conv2d(self.inter_channels, in_channels, 1)
        self.Fw = nn.Conv2d(self.inter_channels, in_channels, 1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 确保最小尺寸
        if H < 2 or W < 2:
            return x
            
        # 计算水平方向池化
        z_h = torch.mean(x, dim=3, keepdim=True)  # [B,C,H,1]
        
        # 计算垂直方向池化
        z_w = torch.mean(x, dim=2, keepdim=True)  # [B,C,1,W]
        z_w = z_w.permute(0, 1, 3, 2)  # [B,C,W,1]
        
        # 拼接
        y = torch.cat([z_h, z_w], dim=2)  # [B,C,(H+W),1]
        
        # 转换
        y = self.F1(y)
        
        # 分离特征
        h_part, w_part = torch.split(y, [H, W], dim=2)  # [B,C',H,1] and [B,C',W,1]
        w_part = w_part.permute(0, 1, 3, 2)  # [B,C',1,W]
        
        # 生成注意力权重
        a_h = self.sigmoid(self.Fh(h_part))  # [B,C,H,1]
        a_w = self.sigmoid(self.Fw(w_part))  # [B,C,1,W]
        
        # 应用注意力
        attention = a_h * a_w  # 广播机制会处理维度
        
        return x * attention



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
            if out.size() != identity.size():
                out = F.interpolate(out, size=identity.size()[2:], mode='bilinear', align_corners=False)
            out = out + identity
        
        return out

class SimulatedConv1(nn.Module):
    """初始卷积层(Conv1)
    将单通道数据转换为多通道特征，模拟数据为图像形式
    """
    def __init__(self, in_channels=1, out_channels=32):
        super().__init__()
        self.in_channels = in_channels
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
    def __init__(self, input_dim, num_classes=13):
        super().__init__()
        self.matrix_size = int(np.ceil(np.sqrt(input_dim)))
        
        # 初始卷积
        self.conv1 = SimulatedConv1(out_channels=32)
        
        # 调整步长和通道数以适应小输入
        self.circulation_configs = [
            # (in_ch, exp_ratio, out_ch, stride)
            (32, 4, 24, 1),  # 降低扩展比例
            (24, 4, 48, 1),
            (48, 4, 72, 1),
            (72, 4, 96, 1)
        ]
        
        # 循环层
        self.circulation_layers = self._make_circulation_layers()
        
        # 坐标注意力
        self.coord_attention = ImprovedCoordinateAttention(96, reduction_ratio=8)  # 增加reduction_ratio
        
        # Conv4
        self.conv4 = RecoveryConv4(96, recovery_ratio=0.75)
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 分类器，添加dropout和中间层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(72, 144),
            nn.ReLU6(),
            nn.Dropout(0.2),
            nn.Linear(144, num_classes)
        )
    
    def _make_circulation_layers(self):
        layers = []
        for in_ch, exp_ratio, out_ch, stride in self.circulation_configs:
            layers.append(CirculationBlock(in_ch, exp_ratio, out_ch, stride))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 输入层
        x = self.conv1(x)
        
        # 循环层
        for layer in self.circulation_layers:
            x = layer(x)
            
        # 注意力机制（如果特征图太小则跳过）
        if x.size(2) >= 2 and x.size(3) >= 2:
            x = self.coord_attention(x)
        
        # 恢复卷积
        x = self.conv4(x)
        
        # 全局池化
        x = self.global_pool(x)
        
        # 分类
        x = self.classifier(x)
        
        return x


# # 测试模型适应动态输入维度
# def test_model(input_dim):
#     """测试模型"""
#     # 创建示例输入
#     batch_size = 2
#     x = torch.randn(batch_size, 1, input_dim, input_dim)
    
#     # 初始化模型
#     model = LMCA(input_dim=input_dim, num_classes=13)
#     model.eval()  # 设置为评估模式
    
#     # 追踪每一层的输出大小
#     with torch.no_grad():
#         output = model(x)
    
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {output.shape}")
    
#     # 计算参数量
#     total_params = sum(p.numel() for p in model.parameters())
#     # print(f"Total parameters: {total_params:,}")

# if __name__ == "__main__":
#     test_model(input_dim=9)  # 测试维度为9
#     test_model(input_dim=16)  # 测试维度为16

# Input shape: torch.Size([2, 1, 9, 9])
# Output shape: torch.Size([2, 13])
# Total parameters: 835,901
# Input shape: torch.Size([2, 1, 16, 16])
# Output shape: torch.Size([2, 13])
# Total parameters: 835,901