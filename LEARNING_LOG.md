跑完 step1_explore_data.py 后，我对 MNIST 数据集有了初步了解：
训练集有 60000 张图片，测试集有 10000 张
每张图片是 28×28 像素的灰度图（只有黑白，没有彩色通道）
各数字的样本数量比较均衡，都在 5400-6700 之间，不存在严重的数据不平衡问题
经过 ToTensor + Normalize 后，像素值从 [0, 255] 变成了 [-1, 1]
MLP（多层感知机）的结构是：
输入(784) → 全连接层(784→256) → ReLU → 全连接层(256→10) → 输出(10)
Batch Size：64（每次送 64 张图给模型）
Epochs：10（整个训练集过 10 遍）
Learning Rate：0.001（每次参数更新的步长）
损失函数：CrossEntropyLoss（交叉熵，分类问题标配）
优化器：Adam（自适应学习率）
最终测试集准确率：97.33%
Loss 随 Epoch 增加逐渐下降，说明模型在不断学习
准确率随 Epoch 增加逐渐上升
训练集准确率高于测试集，说明存在一定程度的过拟合，但不严重
跑 step3_train.py 时报错：ModuleNotFoundError: No module named 'matplotlib'
反思：遇到缺少模块的报错，先检查 requirements.txt 里有没有这个包。如果没有，装上后记得更新 requirements.txt

Level 1 的 MLP 有一个关键弱点：它用 `nn.Flatten()` 把 28×28 的图片展平成了 784 维向量。展平后，原本在图片上相邻的两个像素（比如数字"7"的横和竖），在向量里可能隔得很远。MLP 看不出它们的空间关系，只能死记硬背每个位置的像素值。
CNN 的做法完全不同：它**不展平图片**，保留二维结构。用一个小窗口（卷积核，如 3×3）在图片上滑动，每次只看一小块区域，提取局部特征（边缘、纹理等）。
**CNN 的三大优势**：
- **局部连接**：卷积核只看 3×3 的小区域，符合"相邻像素有关联"的常识
- **权值共享**：同一个卷积核扫过整张图，不管特征在哪里都能被检测到
- **参数更少**：一个 3×3 卷积核只有 9 个权重，而 MLP 的一个全连接层有 784×256 ≈ 20 万个参数
|      指标      |   MLP   |   CNN  |
| 总参数量       | 203,530 | 206,992|
| 最终测试准确率 | 97.60%  | 99.02% |
| 最高测试准确率 | 97.60%  | 99.19% |
| 最终训练 Loss  | 0.0200  | 0.0091 |

--- 训练 MLP on Fashion-MNIST ---
  MLP Epoch 1/5 Test Acc: 82.74%
  MLP Epoch 2/5 Test Acc: 85.70%
  MLP Epoch 3/5 Test Acc: 86.78%
  MLP Epoch 4/5 Test Acc: 86.30%
  MLP Epoch 5/5 Test Acc: 86.67%

--- 训练 CNN on Fashion-MNIST ---
  CNN Epoch 1/5 Test Acc: 86.95%
  CNN Epoch 2/5 Test Acc: 89.16%
  CNN Epoch 3/5 Test Acc: 90.07%
  CNN Epoch 4/5 Test Acc: 90.10%
  CNN Epoch 5/5 Test Acc: 91.18%

Fashion-MNIST 最终准确率对比:
  MLP: 86.67%
  CNN: 91.18%
  CNN 领先: 4.51%  <-- 差距比 MNIST 大得多！
