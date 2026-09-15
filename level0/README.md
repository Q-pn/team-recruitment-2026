## 项目简介
从 MLP → CNN → 经典网络 → U-Net，最终实现手写笔记擦除。

## 硬件与环境
- 显卡：NVIDIA GeForce RTX 5060 Laptop（8GB 显存，Blackwell 架构 sm_120）
- 驱动：577.03（最高支持 CUDA 12.9）
- 系统：Windows 11 + WSL2 (Ubuntu 24.04)
- Python：3.11 / PyTorch：2.11.0+cu128

## 环境复现步骤
conda env create -f environment.yml

## 各 Level 进度
- [x] Level 0：环境管理
- [ ] Level 1：MLP + MNIST
- [ ] Level 2：CNN 对比实验
- [ ] Level 3：AlexNet / ResNet / U-Net
- [ ] Level 4：U-Net 手写笔记擦除
