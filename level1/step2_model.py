"""
Level 1 - Step 2: 搭建 MLP 模型
理解：模型就是把一堆 Linear(线性变换) 和 ReLU(激活函数) 串起来
"""
import torch.nn as nn

class MLP(nn.Module):
    """
    一个简单的手写数字识别网络。

    结构：输入(784) → 隐藏层(256) → ReLU → 输出层(10)

    参数说明：
    - nn.Flatten(): 把 28×28 的图片展平成 784 维向量
    - nn.Linear(784, 256): 全连接层，784 个输入 → 256 个输出
    - nn.ReLU(): 激活函数，把负数变成 0，正数不变
    - nn.Linear(256, 10): 全连接层，256 个输入 → 10 个输出（对应 0-9）
    """
    def __init__(self):
        # 调用父类的 __init__，这是必须的
        super().__init__()

        # nn.Sequential 把多个层串起来，数据按顺序流过每一层
        self.net = nn.Sequential(
            nn.Flatten(),           # [batch, 1, 28, 28] → [batch, 784]
            nn.Linear(784, 256),    # [batch, 784] → [batch, 256]
            nn.ReLU(),              # 激活函数，引入非线性
            nn.Linear(256, 10),     # [batch, 256] → [batch, 10]
        )

    def forward(self, x):
        """
        前向传播：数据从输入到输出的流动过程。
        x 的形状变化：[batch, 1, 28, 28] → [batch, 10]
        """
        return self.net(x)


# 测试模型是否能正常运行
if __name__ == "__main__":
    import torch

    model = MLP()
    print("模型结构:")
    print(model)

    # 数一下总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")

    # 用一个假数据测试前向传播
    # torch.randn(4, 1, 28, 28) 生成 4 张随机的 28×28 图片
    fake_input = torch.randn(4, 1, 28, 28)
    output = model(fake_input)
    print(f"\n输入形状: {fake_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出值(未归一化的分数，即 logits):")
    print(output)
