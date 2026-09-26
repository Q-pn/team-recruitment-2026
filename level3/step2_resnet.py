"""
Level 3 - Step 2: ResNet（残差网络）
核心创新：残差连接（Residual Connection / Skip Connection）
解决深层网络梯度消失问题，让网络可以堆叠到 100+ 层。
"""
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    残差块：ResNet 的基本单元。
    
    数据流：
    输入 x → Conv → ReLU → Conv → 得到 F(x)
    同时 x 通过跳跃连接（skip connection）直接加到 F(x) 上
    最终输出 = F(x) + x，再经过 ReLU
    
    关键理解：网络只需要学习"残差" F(x) = 输出 - 输入，
    如果这几层没什么可学的，F(x) 趋近于 0，输出 ≈ 输入，不会变差。
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)  # 批归一化：稳定训练
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        # 如果输入和输出的通道数或尺寸不同，用 1x1 卷积对齐
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 主路径
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # 跳跃连接：把输入直接加到输出上
        out = out + self.shortcut(x)  # ← 这就是残差连接！
        
        out = self.relu(out)
        return out


class SimpleResNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 初始卷积
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        
        # 残差块堆叠
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
        )
        
        self.layer2 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),  # stride=2 使尺寸减半
            ResidualBlock(64, 64),
        )
        
        self.layer3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
        )
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化，不管输入多大都变成 1x1
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# 测试
if __name__ == "__main__":
    model = SimpleResNet()
    print("ResNet 模型结构:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")
    
    # 追踪形状变化
    x = torch.randn(1, 1, 28, 28)
    print(f"\n输入: {x.shape}")
    x = model.relu(model.bn1(model.conv1(x)))
    print(f"初始卷积后: {x.shape}")
    x = model.layer1(x)
    print(f"layer1 后: {x.shape}")
    x = model.layer2(x)
    print(f"layer2 后: {x.shape}")
    x = model.layer3(x)
    print(f"layer3 后: {x.shape}")
    x = model.avgpool(x)
    print(f"全局平均池化后: {x.shape}")
