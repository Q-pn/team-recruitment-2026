"""
Level 2 - Step 2: 训练 CNN
注意：所有的超参数（Batch Size, Epochs, LR）必须和 Level 1 完全一样！
这就是控制变量法。
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

# ========== 1. 超参数（和 Level 1 一模一样） ==========
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

# ========== 2. 数据准备（和 Level 1 一模一样） ==========
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

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False
)

# ========== 3. 导入 CNN 模型 ==========
from step1_cnn_model import SimpleCNN

model = SimpleCNN()

# ========== 4. 损失函数和优化器（和 Level 1 一模一样） ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ========== 5. 训练循环（和 Level 1 一模一样） ==========
train_losses = []
test_accs = []

print("开始训练 CNN...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (batch_idx + 1) % 200 == 0:
            print(f"  Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(avg_loss)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    test_accs.append(test_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

# ========== 6. 保存模型和结果 ==========
torch.save(model.state_dict(), 'cnn_mnist.pth')
print(f"\n训练完成！CNN 模型已保存到 cnn_mnist.pth")

results = {
    'train_losses': train_losses,
    'test_accs': test_accs
}
with open('cnn_results.json', 'w') as f:
    json.dump(results, f)

# ========== 7. 画 CNN 自己的曲线 ==========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(range(1, EPOCHS+1), train_losses, 'b-', marker='o')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('CNN Training Loss')
ax1.grid(True)

ax2.plot(range(1, EPOCHS+1), test_accs, 'g-', marker='o')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('CNN Test Accuracy')
ax2.grid(True)

plt.tight_layout()
plt.savefig('cnn_training_curves.png', dpi=150)
print("CNN 训练曲线已保存到 cnn_training_curves.png")
