"""
Level 3 - Step 4: 在 MNIST 上训练 AlexNet、ResNet、U-Net(分类版) 并对比
本脚本自包含所有模型定义，不依赖其他 step 文件。
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ========== AlexNet 模型定义 ==========
class SimpleAlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
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


# ========== ResNet 模型定义 ==========
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class SimpleResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ========== U-Net 模型定义 ==========
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, out_channels, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)


# ========== U-Net 分类版 ==========
class UNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet(in_channels=1, out_channels=1)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1, 10)
        )

    def forward(self, x):
        x = self.unet(x)
        x = self.classifier(x)
        return x


# ========== 数据和训练 ==========
BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# U-Net 需要输入尺寸是 16 的倍数，28 不行，用 padding 补到 32
transform_unet = transforms.Compose([
    transforms.Pad(2),  # 28+2+2=32
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("加载 MNIST 数据集...")
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_dataset_unet = torchvision.datasets.MNIST(root='./data', train=True, transform=transform_unet, download=True)
test_dataset_unet = torchvision.datasets.MNIST(root='./data', train=False, transform=transform_unet, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
train_loader_unet = torch.utils.data.DataLoader(train_dataset_unet, batch_size=BATCH_SIZE, shuffle=True)
test_loader_unet = torch.utils.data.DataLoader(test_dataset_unet, batch_size=BATCH_SIZE, shuffle=False)


def quick_train(model, name, train_ld, test_ld, epochs=EPOCHS):
    print(f"\n{'='*50}")
    print(f"训练 {name} on MNIST ({epochs} epochs)")
    print(f"{'='*50}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    test_accs = []

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for images, labels in train_ld:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            _, predicted = model(images).max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        model.eval()
        correct_test, total_test = 0, 0
        with torch.no_grad():
            for images, labels in test_ld:
                outputs = model(images)
                _, predicted = outputs.max(1)
                total_test += labels.size(0)
                correct_test += predicted.eq(labels).sum().item()

        test_acc = 100. * correct_test / total_test
        test_accs.append(test_acc)
        train_acc = 100. * correct / total
        print(f"  Epoch [{epoch+1}/{epochs}] Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")

    return test_accs


# 训练三个模型
alexnet_accs = quick_train(SimpleAlexNet(), "AlexNet", train_loader, test_loader)
resnet_accs = quick_train(SimpleResNet(), "ResNet", train_loader, test_loader)
unet_accs = quick_train(UNetClassifier(), "U-Net(Classifier)", train_loader_unet, test_loader_unet)

# 对比
print(f"\n{'='*50}")
print(f"{'网络':<20} {'最终准确率':>12} {'参数量':>14}")
print(f"{'-'*50}")

models = [
    ("AlexNet", SimpleAlexNet(), alexnet_accs),
    ("ResNet", SimpleResNet(), resnet_accs),
    ("U-Net(Classifier)", UNetClassifier(), unet_accs),
]

for name, model, accs in models:
    params = sum(p.numel() for p in model.parameters())
    print(f"{name:<20} {accs[-1]:>11.2f}% {params:>14,}")

# 画图
fig, ax = plt.subplots(figsize=(8, 5))
epochs_range = range(1, EPOCHS + 1)
ax.plot(epochs_range, alexnet_accs, 'b-o', label='AlexNet')
ax.plot(epochs_range, resnet_accs, 'g-o', label='ResNet')
ax.plot(epochs_range, unet_accs, 'r-o', label='U-Net(Classifier)')
ax.set_xlabel('Epoch')
ax.set_ylabel('Test Accuracy (%)')
ax.set_title('Classic Networks on MNIST')
ax.legend()
ax.grid(True)
plt.savefig('classic_networks_comparison.png', dpi=150)
print("\n对比图已保存到 classic_networks_comparison.png")
