"""Dual-Branch Perception Module (Remote Sensing 2026, 18, 839, Fig. 2 and Eqs. 1-4)."""

import torch
import torch.nn as nn

__all__ = ("DBPM",)


class DBPM(nn.Module):
    """Combine square and strip receptive fields with spatial branch selection.

    Args:
        in_channels (int): Input channels, summed for a list of input features.
        out_channels (int): Output channels.
        k_strip (int): Odd kernel length for the horizontal and vertical strips.

    The paper specifies the topology and half-channel branch projections, but
    does not specify kernel sizes, dilation or the spatial transformation kernel.
    Defaults follow this project's original full MS_LKA design: 5x5 depthwise,
    7x7 depthwise with dilation 3, 1x7 then 7x1 depthwise strips, and a 7x7
    spatial selection convolution producing two independent sigmoid weights.

    Like MSAA, this accepts a tensor or same-resolution feature list. An input
    projection adapts unequal channel counts; it is Identity for the paper's
    channel-preserving case. No activation is added after the restored attention
    map, and no residual addition is used, following Eq. (4).
    """

    def __init__(self, in_channels, out_channels, k_strip=7):
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError("in_channels and out_channels must be positive")
        if not isinstance(k_strip, int) or k_strip < 1 or k_strip % 2 == 0:
            raise ValueError("k_strip must be a positive odd integer")

        dim = out_channels
        hidden_channels = max(1, dim // 2)
        strip_padding = k_strip // 2

        # MSAA 风格的输入输出通道接口；通道相同时严格保持论文的输入路径。
        self.input_proj = nn.Identity() if in_channels == dim else nn.Conv2d(in_channels, dim, 1)

        # 1. 并行感知：大核方形分支和串行水平/垂直条带分支，式 (1)。
        self.conv_square = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=2, groups=dim),
            nn.Conv2d(dim, dim, 7, padding=9, dilation=3, groups=dim),
        )
        self.conv_strip = nn.Sequential(
            nn.Conv2d(dim, dim, (1, k_strip), padding=(0, strip_padding), groups=dim),
            nn.Conv2d(dim, dim, (k_strip, 1), padding=(strip_padding, 0), groups=dim),
        )

        # 2. 两个分支分别投影至一半通道，式 (2)。
        self.conv_proj_square = nn.Conv2d(dim, hidden_channels, 1)
        self.conv_proj_strip = nn.Conv2d(dim, hidden_channels, 1)

        # 3. 沿通道聚合空间统计，生成两张分支选择权重图，式 (3)。
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

        # 4. 恢复通道得到注意力图，式 (4)。
        self.conv_m = nn.Conv2d(hidden_channels, dim, 1)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = torch.cat(x, dim=1)
        x = self.input_proj(x)

        feat_square = self.conv_proj_square(self.conv_square(x))
        feat_strip = self.conv_proj_strip(self.conv_strip(x))

        features = torch.cat([feat_square, feat_strip], dim=1)
        avg_attn = torch.mean(features, dim=1, keepdim=True)
        max_attn = torch.max(features, dim=1, keepdim=True)[0]
        weights = self.sigmoid(self.conv_squeeze(torch.cat([avg_attn, max_attn], dim=1)))

        fused = feat_square * weights[:, 0:1] + feat_strip * weights[:, 1:2]
        attention = self.conv_m(fused)
        return x * attention


if __name__ == "__main__":
    input_tensor = torch.randn(1, 512, 7, 7)
    model = DBPM(in_channels=512, out_channels=512)
    print(model(input_tensor).shape)
