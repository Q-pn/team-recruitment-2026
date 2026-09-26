"""
Level 2 - Step 3: 对比 MLP 和 CNN
读取 Level 1 和 Level 2 的结果，画出对比曲线，打印对比表格。
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn

# ========== 1. 统计参数量 ==========
# MLP 的结构
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

mlp_params = sum(p.numel() for p in MLP().parameters())
cnn_params = sum(p.numel() for p in SimpleCNN().parameters())

# ========== 2. 读取训练结果 ==========
# 读取 Level 1 的 MLP 结果（注意路径是 ../level1/）
try:
    with open('../level1/mlp_results.json', 'r') as f:
        mlp_data = json.load(f)
except FileNotFoundError:
    print("找不到 Level 1 的 mlp_results.json！")
    print("请确保你在 Level 1 的 step3_train.py 中保存了结果，或者手动把 Level 1 的 test_accs 填到下面。")
    # 如果找不到文件，这里给一个 Level 1 的典型数据作为备用
    mlp_data = {
        "train_losses": [0.31, 0.16, 0.11, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02],
        "test_accs": [93.5, 95.1, 96.0, 96.5, 96.8, 97.0, 97.2, 97.4, 97.5, 97.6]
    }

with open('cnn_results.json', 'r') as f:
    cnn_data = json.load(f)

epochs = range(1, 11)

# ========== 3. 打印对比表格 ==========
print("=" * 50)
print(f"{'指标':<20} | {'MLP':<12} | {'CNN':<12}")
print("-" * 50)
print(f"{'总参数量':<18} | {mlp_params:<12,} | {cnn_params:<12,}")
print(f"{'最终测试准确率':<16} | {mlp_data['test_accs'][-1]:<11.2f}% | {cnn_data['test_accs'][-1]:<11.2f}%")
print(f"{'最高测试准确率':<16} | {max(mlp_data['test_accs']):<11.2f}% | {max(cnn_data['test_accs']):<11.2f}%")
print(f"{'最终训练 Loss':<16} | {mlp_data['train_losses'][-1]:<12.4f} | {cnn_data['train_losses'][-1]:<12.4f}")
print("=" * 50)

# ========== 4. 画对比图 ==========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 图1: Loss 对比
ax1.plot(epochs, mlp_data['train_losses'], 'b-o', label='MLP')
ax1.plot(epochs, cnn_data['train_losses'], 'r-o', label='CNN')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss')
ax1.set_title('Training Loss: MLP vs CNN')
ax1.legend()
ax1.grid(True)

# 图2: Accuracy 对比
ax2.plot(epochs, mlp_data['test_accs'], 'b-o', label='MLP')
ax2.plot(epochs, cnn_data['test_accs'], 'r-o', label='CNN')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Test Accuracy (%)')
ax2.set_title('Test Accuracy: MLP vs CNN')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('mlp_vs_cnn_comparison.png', dpi=150)
print("\n对比图已保存到 mlp_vs_cnn_comparison.png")
print("请把这张图截图放进你的 README 或学习文档中！")
