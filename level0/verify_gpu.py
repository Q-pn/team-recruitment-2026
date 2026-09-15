"""
Level 0 - GPU 环境验证脚本
用途：确认 PyTorch 能正确调用 NVIDIA GPU 进行计算
"""
import torch

print("=" * 50)
print("PyTorch 环境检查")
print("=" * 50)

print(f"PyTorch 版本   : {torch.__version__}")
print(f"CUDA 版本      : {torch.version.cuda}")
print(f"cuDNN 版本     : {torch.backends.cudnn.version()}")
print(f"CUDA 是否可用  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU 数量       : {torch.cuda.device_count()}")
    print(f"GPU 型号       : {torch.cuda.get_device_name(0)}")
    print(f"显卡算力       : {torch.cuda.get_device_capability(0)}")

    # 关键：真正做一次 GPU 计算，而不只是查询状态
    # is_available() 为 True 不代表 kernel 能跑（sm_120 兼容性问题的典型陷阱）
    x = torch.randn(1000, 1000, device="cuda")
    y = torch.randn(1000, 1000, device="cuda")
    z = x @ y  # 矩阵乘法，实际调用 CUDA kernel
    torch.cuda.synchronize()  # 等待 GPU 计算完成

    print(f"GPU 矩阵运算   : 成功，结果形状 {tuple(z.shape)}")
    print(f"显存占用       : {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    print("=" * 50)
    print("环境验证通过，可以进入 Level 1")
else:
    print("=" * 50)
    print("CUDA 不可用，请检查：")
    print("  1. Windows 侧 NVIDIA 驱动是否已安装（在 Windows 里跑 nvidia-smi）")
    print("  2. WSL 里跑 nvidia-smi 是否能看到显卡")
    print("  3. 装的 PyTorch 是否是 cu128 版本（不是 cpu 版）")
