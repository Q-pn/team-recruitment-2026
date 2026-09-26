"""
Level 2 - Step 4: 在更难的 Fashion-MNIST 上对比 MLP 和 CNN
Fashion-MNIST 包含 10 类服饰图片（T恤、裤子、鞋子等），比手写数字难得多。
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ========== 1. 数据准备（注意用的是 FashionMNIST） ==========
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# ========== 2. 定义两个模型 ==========
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
    def forward(self, x):
        return self.net(x)

from step1_cnn_model import SimpleCNN

# ========== 3. 快速训练函数（只跑 5 个 Epoch 节省时间） ==========
def quick_train(model, name, epochs=5):
    print(f"\n--- 训练 {name} on Fashion-MNIST ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    test_accs = []

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        # 测试
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        test_accs.append(acc)
        print(f"  {name} Epoch {epoch+1}/{epochs} Test Acc: {acc:.2f}%")
    
    return test_accs

# ========== 4. 训练并对比 ==========
mlp_accs = quick_train(MLP(), "MLP", epochs=5)
cnn_accs = quick_train(SimpleCNN(), "CNN", epochs=5)

print(f"\n{'='*50}")
print(f"Fashion-MNIST 最终准确率对比:")
print(f"  MLP: {mlp_accs[-1]:.2f}%")
print(f"  CNN: {cnn_accs[-1]:.2f}%")
print(f"  CNN 领先: {cnn_accs[-1] - mlp_accs[-1]:.2f}%  <-- 差距比 MNIST 大得多！")
print(f"{'='*50}")

# 画图
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, 6), mlp_accs, 'b-o', label='MLP')
ax.plot(range(1, 6), cnn_accs, 'r-o', label='CNN')
ax.set_xlabel('Epoch')
ax.set_ylabel('Test Accuracy (%)')
ax.set_title('Fashion-MNIST: MLP vs CNN')
ax.legend()
ax.grid(True)
plt.savefig('fashion_mnist_comparison.png', dpi=150)
print("对比图已保存到 fashion_mnist_comparison.png")
