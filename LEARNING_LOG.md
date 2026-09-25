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
