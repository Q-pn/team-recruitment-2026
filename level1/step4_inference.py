"""
Level 1 - Step 4: 单张图片推理
用训练好的模型识别一张新的手写数字图片
"""
import torch
import torchvision
import torchvision.transforms as transforms
from step2_model import MLP
from PIL import Image

# 1. 加载训练好的模型
model = MLP()
model.load_state_dict(torch.load('mlp_mnist.pth', weights_only=True))
model.eval()  # 切换到评估模式

# 2. 准备数据变换（必须和训练时一模一样）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 3. 从测试集中随机取一张图片来识别
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transform
)

# 取第 42 张（可以换成任意索引）
idx = 42
img, true_label = test_dataset[idx]

# 4. 推理（预测）
with torch.no_grad():
    # model 期望输入是 [batch, channel, height, width]
    # 单张图片需要加一个 batch 维度：[1, 1, 28, 28]
    output = model(img.unsqueeze(0))

    # softmax 把输出变成概率（所有类别的概率加起来 = 1）
    probs = torch.softmax(output, dim=1)

    # 取概率最高的
    predicted = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted].item()

print(f"图片索引: {idx}")
print(f"真实标签: {true_label}")
print(f"模型预测: {predicted}")
print(f"置信度:   {confidence:.2%}")

if predicted == true_label:
    print("✓ 预测正确！")
else:
    print("✗ 预测错误！")

# 打印每个数字的概率
print(f"\n各类别概率:")
for digit in range(10):
    bar = "█" * int(probs[0][digit].item() * 50)
    print(f"  {digit}: {probs[0][digit].item():.4f} {bar}")
