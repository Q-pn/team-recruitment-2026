"""
Level 1 - Step 1: 数据探索
目的：看看 MNIST 数据集长什么样，建立直观感受
"""
import torchvision
import torchvision.transforms as transforms
import torch

# 1. 定义数据变换：把图片变成 Tensor（张量），并归一化到 [0,1]
#    transforms.ToTensor() 做两件事：
#    a) 把 PIL Image 转成 Tensor
#    b) 像素值从 [0, 255] 缩放到 [0.0, 1.0]
#    transforms.Normalize((0.5,), (0.5,)) 再做一步：
#    把 [0, 1] 映射到 [-1, 1]，让网络更容易学习
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 2. 下载训练集和测试集
#    root='./data' 表示数据保存在当前目录的 data 文件夹里
#    train=True/False 区分训练集和测试集
#    download=True 如果本地没有就自动下载
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transform, download=True
)

# 3. 看看数据集的基本信息
print(f"训练集大小: {len(train_dataset)} 张图片")
print(f"测试集大小: {len(test_dataset)} 张图片")

# 4. 取出第一张图片看看
img, label = train_dataset[0]
print(f"\n第一张图片的信息:")
print(f"  标签(真实答案): {label}")
print(f"  Tensor 形状: {img.shape}")    # 应该是 [1, 28, 28]
print(f"  含义: 1个通道(灰度图), 28行, 28列")
print(f"  像素值范围: [{img.min():.2f}, {img.max():.2f}]")

# 5. 看看所有标签的分布（每个数字有多少张）
from collections import Counter
labels = [train_dataset[i][1] for i in range(len(train_dataset))]
counts = Counter(labels)
print(f"\n各数字的样本数量:")
for digit in sorted(counts.keys()):
    print(f"  数字 {digit}: {counts[digit]} 张")
