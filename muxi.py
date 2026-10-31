#!/usr/bin/env python3
"""
沐兮 (MuXi) - 你的 AI 模型
基于 PyTorch 的深度学习框架
"""

print("="*60)
print("沐兮 (MuXi) AI 框架")
print("="*60)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # 检查 AMD GPU
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✓ 检测到 AMD GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print(f"✓ 使用 CPU 模式")

    print("✓ PyTorch 已就绪")
    PYTORCH_AVAILABLE = True

except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️  PyTorch 未安装")
    print("\n安装命令:")
    print("  sudo pacman -S python-pytorch-rocm python-torchvision-rocm")
    print("\n沐兮的架构已准备好，等待 PyTorch...")
    print("="*60)

if not PYTORCH_AVAILABLE:
    exit(0)

print("="*60)


class MuXi(nn.Module):
    """
    沐兮 - 你的专属 AI 模型
    一个灵活的神经网络架构
    """

    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.2):
        """
        初始化沐兮

        参数:
            input_size: 输入特征数量
            hidden_sizes: 隐藏层大小列表，例如 [64, 32]
            output_size: 输出大小
            dropout: Dropout 比例
        """
        super(MuXi, self).__init__()

        self.name = "沐兮 (MuXi)"
        self.version = "1.0"

        # 构建网络层
        layers = []
        prev_size = input_size

        # 隐藏层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # 输出层
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)
        self.param_count = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        """前向传播"""
        return self.network(x)

    def info(self):
        """显示模型信息"""
        print(f"\n{'='*60}")
        print(f"模型名称: {self.name}")
        print(f"版本: {self.version}")
        print(f"参数量: {self.param_count:,}")
        print(f"{'='*60}")
        print(f"网络结构:")
        print(self.network)
        print(f"{'='*60}\n")


class MuXiTrainer:
    """沐兮的训练器"""

    def __init__(self, model, learning_rate=0.001, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.history = {'train_loss': [], 'val_loss': []}

    def train_epoch(self, train_loader):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0

        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader=None, epochs=100, verbose=True):
        """训练沐兮"""
        print(f"\n🚀 开始训练 {self.model.name}...")
        print(f"设备: {self.device}")
        print(f"训练轮数: {epochs}")
        print(f"{'='*60}\n")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)

            if val_loader:
                val_loss = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                if val_loader:
                    print(f"Epoch {epoch+1:4d}: 训练损失={train_loss:.6f}, 验证损失={val_loss:.6f}")
                else:
                    print(f"Epoch {epoch+1:4d}: 训练损失={train_loss:.6f}")

        print(f"\n✓ 训练完成！")

    def save(self, filepath):
        """保存沐兮"""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'history': self.history,
        }, filepath)
        print(f"✓ 沐兮已保存到: {filepath}")

    def load(self, filepath):
        """加载沐兮"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.history = checkpoint['history']
        print(f"✓ 沐兮已加载: {filepath}")


# ============================================================
# 示例
# ============================================================

if __name__ == "__main__":
    print("\n创建沐兮模型...")

    # 创建沐兮
    muxi = MuXi(
        input_size=10,
        hidden_sizes=[64, 32, 16],
        output_size=3,
        dropout=0.2
    )

    # 显示信息
    muxi.info()

    print("沐兮已准备就绪！\n")
    print("使用示例:")
    print("  from muxi import MuXi, MuXiTrainer")
    print("  muxi = MuXi(10, [64, 32], 3)")
    print("  trainer = MuXiTrainer(muxi)")
    print("  trainer.train(train_loader, val_loader, epochs=100)")
    print("  trainer.save('muxi_model.pth')")
    print("="*60)
