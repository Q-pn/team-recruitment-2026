### Level 1: MLP 手写数字识别
- 模型：MLP（784 → 256 → 10）
- 数据集：MNIST
- 测试集准确率：97.33%
- 包含数据探索、模型定义、训练、推理、错误分析
### Level 2: CNN 对比实验
- 模型：SimpleCNN (Conv2d + MaxPool2d x 2 + FC)
- 控制变量：Batch=64, Epoch=10, LR=0.001 (与 Level 1 完全一致)
- MNIST 结果：MLP 97.6% vs CNN 99.2% (CNN 更高且参数更少)
- Fashion-MNIST 结果：MLP 86.67% vs CNN 91.18% (复杂图像上 CNN 优势显著)

