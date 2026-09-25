"""
Level 2 - Step 1: 搭建 CNN 模型
理解卷积层（Conv2d）、激活函数（ReLU）和池化层（MaxPool2d）的作用。
"""
import torch.nn as nn
import torch

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 特征提取部分：用卷积和池化提取图片的局部特征
        self.features = nn.Sequential(
            # 第1个卷积块
            # Conv2d(输入通道, 输出通道, 卷积核大小, padding)
            # 输入: 1个通道(灰度图), 28x28
            # 输出: 16个通道(16种不同的特征图), 28x28 (padding=1 保持了尺寸)
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            # 池化层：把图片缩小一半，只保留每个 2x2 区域里的最大值
            # 作用：减少计算量，同时让特征具有"平移不变性"
            # 输出: 16个通道, 14x14
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第2个卷积块
            # 输入: 16个通道, 14x14
            # 输出: 32个通道, 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # 输出: 32个通道, 7x7
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 分类部分：和 MLP 一样，用全连接层输出 10 个类别
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 把 32x7x7 展平成 32*7*7 = 1568
            nn.Linear(1568, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)    # 先提取特征
        x = self.classifier(x)  # 再分类
        return x

# 测试模型
if __name__ == "__main__":
    model = SimpleCNN()
    print("CNN 模型结构:")
    print(model)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nCNN 总参数量: {total_params:,}")
    print(f"(对比 Level 1 的 MLP 参数量: 203,530)")

    # 用一个假数据测试形状变化
    fake_input = torch.randn(64, 1, 28, 28) # 模拟一个 batch 的 64 张图片
    print(f"\n输入形状: {fake_input.shape}")
    
    # 逐步看形状变化
    x = model.features[0](fake_input)
    print(f"经过 Conv2d(1, 16, 3, padding=1) 后: {x.shape}")
    x = model.features[1](x)
    print(f"经过 ReLU 后: {x.shape}")
    x = model.features[2](x)
    print(f"经过 MaxPool2d(2) 后: {x.shape}  <-- 尺寸减半了！")
    
    output = model(fake_input)
    print(f"\n最终输出形状: {output.shape}")
