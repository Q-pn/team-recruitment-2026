"""
Level 3 - Step 1: AlexNet
2012 年 ImageNet 冠军网络，开启了深度学习时代。
核心创新：深层 CNN + ReLU 激活 + Dropout + 数据增强

原始 AlexNet 是为 224x224 的彩色图片设计的，这里我们修改为适配 28x28 的 MNIST。
"""
import torch
import torch.nn as nn

class SimpleAlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 特征提取：5 层卷积
        self.features = nn.Sequential(
            # 第1层卷积
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 1→32 (原版64太多，MNIST用32足够)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 28→14
            
            # 第2层卷积
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32→64
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 14→7
            
            # 第3层卷积
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 64→128
            nn.ReLU(),
            
            # 第4层卷积
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # 第5层卷积
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 7→3
        )
        
        # 分类器：3 层全连接 + Dropout
        self.classifier = nn.Sequential(
            nn.Flatten(),                                  # 64*3*3 = 576
            nn.Dropout(0.5),                               # Dropout: 训练时随机丢弃 50% 的神经元
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 测试
if __name__ == "__main__":
    model = SimpleAlexNet()
    print("AlexNet 模型结构:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")
    
    fake_input = torch.randn(4, 1, 28, 28)
    output = model(fake_input)
    print(f"\n输入形状: {fake_input.shape}")
    print(f"输出形状: {output.shape}")
