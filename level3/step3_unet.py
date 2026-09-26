"""
Level 3 - Step 3: U-Net（编码器-解码器 + 跳跃连接）
Level 4 的核心网络！必须理解透彻。

U-Net 结构（像字母 U）：
编码器（缩小） → 最底层 → 解码器（放大）
     ↕ 跳跃连接 ↕

编码器：提取"是什么"的特征（语义信息）
解码器：恢复"在哪里"的位置信息（空间信息）
跳跃连接：把编码器的细节特征直接拼接到解码器，弥补下采样丢失的空间信息
"""
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """U-Net 的基本模块：两层卷积 + BN + ReLU"""
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
        """
        in_channels: 输入图片的通道数（灰度图=1，彩色图=3）
        out_channels: 输出图片的通道数（Level 4 擦除任务输出也是灰度图=1）
        """
        super().__init__()
        
        # ========== 编码器（下采样路径） ==========
        # 每一层：DoubleConv → 保存特征给跳跃连接 → MaxPool 缩小
        self.enc1 = DoubleConv(in_channels, 64)    # 输入 → 64通道
        self.enc2 = DoubleConv(64, 128)             # 64 → 128
        self.enc3 = DoubleConv(128, 256)            # 128 → 256
        self.enc4 = DoubleConv(256, 512)            # 256 → 512
        
        # 最底层
        self.bottleneck = DoubleConv(512, 1024)     # 512 → 1024
        
        # ========== 解码器（上采样路径） ==========
        # 每一层：上采样 → 拼接跳跃连接的特征 → DoubleConv
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)  # 1024→512，尺寸×2
        self.dec4 = DoubleConv(1024, 512)  # 512(上采样) + 512(跳跃连接) = 1024 → 512
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)   # 256 + 256 = 512 → 256
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)   # 128 + 128 = 256 → 128
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)    # 64 + 64 = 128 → 64
        
        # 最终输出层
        self.final = nn.Conv2d(64, out_channels, 1)  # 1x1 卷积调整通道数
        
        # 池化层（编码器用）
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # ========== 编码器 ==========
        e1 = self.enc1(x)          # [B, 64, H, W]    ← 保存给跳跃连接
        e2 = self.enc2(self.pool(e1))  # [B, 128, H/2, W/2]
        e3 = self.enc3(self.pool(e2))  # [B, 256, H/4, W/4]
        e4 = self.enc4(self.pool(e3))  # [B, 512, H/8, W/8]
        
        # ========== 最底层 ==========
        b = self.bottleneck(self.pool(e4))  # [B, 1024, H/16, W/16]
        
        # ========== 解码器 ==========
        # 上采样 + 拼接跳跃连接 + DoubleConv
        d4 = self.up4(b)           # [B, 512, H/8, W/8]
        d4 = torch.cat([d4, e4], dim=1)  # 拼接！[B, 1024, H/8, W/8]  ← 跳跃连接
        d4 = self.dec4(d4)         # [B, 512, H/8, W/8]
        
        d3 = self.up3(d4)          # [B, 256, H/4, W/4]
        d3 = torch.cat([d3, e3], dim=1)  # 拼接！[B, 512, H/4, W/4]
        d3 = self.dec3(d3)         # [B, 256, H/4, W/4]
        
        d2 = self.up2(d3)          # [B, 128, H/2, W/2]
        d2 = torch.cat([d2, e2], dim=1)  # 拼接！[B, 256, H/2, W/2]
        d2 = self.dec2(d2)         # [B, 128, H/2, W/2]
        
        d1 = self.up1(d2)          # [B, 64, H, W]
        d1 = torch.cat([d1, e1], dim=1)  # 拼接！[B, 128, H, W]
        d1 = self.dec1(d1)         # [B, 64, H, W]
        
        # 最终输出
        out = self.final(d1)       # [B, out_channels, H, W]
        
        return out


# 测试
if __name__ == "__main__":
    model = UNet(in_channels=1, out_channels=1)
    print("U-Net 模型结构:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")
    
    # 用 MNIST 尺寸测试（28x28，但 U-Net 需要 16 的倍数，所以用 32x32）
    # 实际 Level 4 会用更大的图片
    fake_input = torch.randn(1, 1, 32, 32)
    print(f"\n输入形状: {fake_input.shape}")
    output = model(fake_input)
    print(f"输出形状: {output.shape}")
    print("\n输入和输出尺寸相同！这就是图像到图像任务的特点。")
    
    # 追踪跳跃连接的数据流
    print("\n--- 编码器-解码器数据流追踪 ---")
    x = torch.randn(1, 1, 32, 32)
    print(f"输入:           {x.shape}")
    
    e1 = model.enc1(x)
    print(f"enc1 (保存→跳跃): {e1.shape}")
    e2 = model.enc2(model.pool(e1))
    print(f"enc2 (保存→跳跃): {e2.shape}")
    e3 = model.enc3(model.pool(e2))
    print(f"enc3 (保存→跳跃): {e3.shape}")
    e4 = model.enc4(model.pool(e3))
    print(f"enc4 (保存→跳跃): {e4.shape}")
    
    b = model.bottleneck(model.pool(e4))
    print(f"bottleneck:       {b.shape}")
    
    d4 = model.up4(b)
    d4_cat = torch.cat([d4, e4], dim=1)
    print(f"dec4 拼接后:      {d4_cat.shape}  ← e4({e4.shape}) + 上采样({d4.shape})")
