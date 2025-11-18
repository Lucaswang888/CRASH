import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channel * 2, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        b, c, _, _ = x.size()
        avg_y = self.avg_pool(x).view(b, c)  # [B, C]
        max_y = self.max_pool(x).view(b, c)  # [B, C]

        y = torch.cat((avg_y, max_y), dim=1)  # [B, 2C]
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


class SpectralGatingNetwork(nn.Module):
    """
    频域 gating 模块：
    - 输入:
        1) [B, dim]
        2) 或 [B, L, dim]（比如 L=20，会先在 L 维上做平均）
    - 内部:
        对特征维做 1D FFT → 频域 gating → iFFT → ChannelAttention
    - 输出: [B, dim]
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # 线性投影 (可选，增加表达能力)
        self.proj_in = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)

        # 频域门控参数：最后一维 2 表示 (real, imag)
        freq_dim = dim // 2 + 1
        self.freq_gate = nn.Parameter(torch.randn(freq_dim, 2) * 0.02)

        # 通道注意力（时域）
        self.channel_attention = ChannelAttention(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        支持两种输入：
        - [B, dim]
        - [B, L, dim]
        """
        if x.dim() == 3:
            # [B, L, C] → 在 L 上做平均，得到全局特征
            B, L, C = x.shape
            x = x.mean(dim=1)  # [B, C]
        elif x.dim() == 2:
            B, C = x.shape
        else:
            raise RuntimeError(f"SpectralGatingNetwork expects 2D or 3D input, got {x.shape}")

        B, C = x.shape
        assert C == self.dim, f"Expected dim={self.dim}, but got {C}"

        # 线性投影
        x = self.proj_in(x)  # [B, dim]

        # 1D FFT 到频域（对 feature 维做 FFT）
        x_freq = torch.fft.rfft(x, dim=1)  # [B, dim//2 + 1]

        # 频域门控
        gate = torch.view_as_complex(self.freq_gate)  # [freq_dim] (complex)
        gate = gate[: x_freq.size(1)]                 # 安全裁剪
        x_freq = x_freq * gate                        # [B, freq_dim]

        # iFFT 回到时域
        x_time = torch.fft.irfft(x_freq, n=C, dim=1)  # [B, dim]

        # 通道注意力需要 4D，暂时 reshape 为 [B, C, 1, 1]
        x_ca = x_time.view(B, C, 1, 1)
        x_ca = self.channel_attention(x_ca)           # [B, C, 1, 1]
        x = x_ca.view(B, C)                           # [B, dim]

        # 输出线性层
        x = self.proj_out(x)                          # [B, dim]
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试 2D 输入
    block = SpectralGatingNetwork(512).to(device)
    inp2 = torch.rand(10, 512).to(device)
    out2 = block(inp2)
    print("2D input shape :", inp2.shape, "→ output:", out2.shape)

    # 测试 3D 输入（比如 [B, 20, 512]）
    inp3 = torch.rand(10, 20, 512).to(device)
    out3 = block(inp3)
    print("3D input shape :", inp3.shape, "→ output:", out3.shape)
