"""
Level 1 - Step 5: 错误分析
找出模型识别错误的图片，看看它容易在哪里犯错
"""
import torch
import torchvision
import torchvision.transforms as transforms
from step2_model import MLP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. 加载模型和数据
model = MLP()
model.load_state_dict(torch.load('mlp_mnist.pth', weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transform
)

# 2. 逐张预测，记录错误
errors = []  # 存放 (图片索引, 真实标签, 预测标签)

with torch.no_grad():
    for i in range(len(test_dataset)):
        img, true_label = test_dataset[i]
        output = model(img.unsqueeze(0))
        predicted = torch.argmax(output, dim=1).item()
        if predicted != true_label:
            errors.append((i, true_label, predicted))

print(f"测试集共 {len(test_dataset)} 张图片")
print(f"预测错误 {len(errors)} 张")
print(f"准确率: {100 * (len(test_dataset) - len(errors)) / len(test_dataset):.2f}%")

# 3. 展示前 16 个错误样本
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
fig.suptitle('Misclassified Examples (first 16)', fontsize=16)

for idx, ax in enumerate(axes.flat):
    if idx >= len(errors):
        break
    img_idx, true_label, pred_label = errors[idx]
    img, _ = test_dataset[img_idx]

    # 反归一化：[-1, 1] → [0, 1] 用于显示
    img_display = img.squeeze() * 0.5 + 0.5

    ax.imshow(img_display, cmap='gray')
    ax.set_title(f'True: {true_label}, Pred: {pred_label}',
                 color='red', fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('error_analysis.png', dpi=150)
print("\n错误样本图已保存到 error_analysis.png")

# 4. 统计最容易混淆的数字对
from collections import Counter
confusion_pairs = Counter([(true, pred) for _, true, pred in errors])
print(f"\n最容易混淆的数字对 (真实→预测: 次数):")
for (true, pred), count in confusion_pairs.most_common(10):
    print(f"  {true} → {pred}: {count} 次") 
