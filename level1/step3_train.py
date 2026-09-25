"""
Level 1 - Step 3: 训练 MLP
这是整个 Level 1 最重要的脚本。训练循环是深度学习的核心流程。
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')  # 无 GUI 环境也能画图
import matplotlib.pyplot as plt

# ========== 1. 超参数（训练前设定好的"旋钮"） ==========
BATCH_SIZE = 64       # 每次送 64 张图片给模型（太大显存不够，太小训练慢）
EPOCHS = 10           # 整个训练集过 10 遍
LEARNING_RATE = 0.001 # 学习率：每次参数更新的步长

# ========== 2. 数据准备 ==========
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transform, download=True
)

# DataLoader: 数据"传送带"
# - batch_size: 每次取多少张
# - shuffle=True: 每个 epoch 开始时打乱顺序（避免模型记住顺序）
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False
)

# ========== 3. 模型、损失函数、优化器 ==========
# 复用 step2 的模型定义
from step2_model import MLP

model = MLP()

# 损失函数：交叉熵损失（分类问题的标配）
# 它把 softmax（把输出变成概率）和负对数似然合在一起算
criterion = nn.CrossEntropyLoss()

# 优化器：Adam（自适应学习率，比 SGD 更省心）
# model.parameters() 告诉优化器"你要调的就是这些参数"
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ========== 4. 训练循环 ==========
train_losses = []    # 记录每个 epoch 的平均训练损失
test_accs = []       # 记录每个 epoch 的测试准确率

for epoch in range(EPOCHS):
    # ---------- 训练阶段 ----------
    model.train()  # 开启训练模式（某些层如 Dropout 在训练和测试时行为不同）
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        # 训练循环四步曲（最重要！背下来）

        # 第一步：清空梯度
        # 为什么？因为 PyTorch 默认会累加梯度，不清空就会把上次的梯度也加进来
        optimizer.zero_grad()

        # 第二步：前向传播
        # 图片送进模型，得到预测结果（10 个分数）
        outputs = model(images)

        # 第三步：计算损失
        # CrossEntropyLoss 会自动处理 labels（不需要 one-hot 编码）
        loss = criterion(outputs, labels)

        # 第四步：反向传播 + 更新参数
        # loss.backward() 计算每个参数的梯度（它对损失的影响有多大）
        # optimizer.step() 根据梯度更新参数（朝损失减小的方向走一小步）
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)  # 取分数最高的类别
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 每 200 个 batch 打印一次进度
        if (batch_idx + 1) % 200 == 0:
            print(f"  Epoch [{epoch+1}/{EPOCHS}], "
                  f"Batch [{batch_idx+1}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f}")

    # 记录本 epoch 的平均损失和训练准确率
    avg_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(avg_loss)

    # ---------- 测试阶段 ----------
    model.eval()  # 开启评估模式
    correct = 0
    total = 0

    with torch.no_grad():  # 测试时不需要计算梯度（省内存、加速）
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    test_accs.append(test_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}] | "
          f"Train Loss: {avg_loss:.4f} | "
          f"Train Acc: {train_acc:.2f}% | "
          f"Test Acc: {test_acc:.2f}%")

# ========== 5. 保存模型 ==========
torch.save(model.state_dict(), 'mlp_mnist.pth')
print(f"\n训练完成！模型已保存到 mlp_mnist.pth")

# ========== 6. 画 Loss 和 Accuracy 曲线 ==========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(range(1, EPOCHS+1), train_losses, 'b-', marker='o')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.grid(True)

ax2.plot(range(1, EPOCHS+1), test_accs, 'g-', marker='o')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Test Accuracy')
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
print("Loss/Accuracy 曲线已保存到 training_curves.png")
