# 沐兮 (MuXi) - AI 模型框架

沐兮是一个基于 PyTorch 的灵活深度学习框架。

## 特性

- ✨ 灵活的网络架构设计
- 🚀 支持 AMD GPU 加速 (ROCm)
- 📊 内置训练器和历史记录
- 💾 模型保存和加载
- 🎯 易于使用的 API

## 安装依赖

```bash
sudo pacman -S python-pytorch-rocm python-torchvision-rocm
```

## 快速开始

```python
from muxi import MuXi, MuXiTrainer

# 创建沐兮模型
muxi = MuXi(
    input_size=10,
    hidden_sizes=[64, 32, 16],
    output_size=3
)

# 创建训练器
trainer = MuXiTrainer(muxi, learning_rate=0.001)

# 训练模型
trainer.train(train_loader, val_loader, epochs=100)

# 保存模型
trainer.save('muxi_model.pth')
```

## 作者

创建于 2024
