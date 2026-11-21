import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv

class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        # x: [Batch, Channel, Height, Width]
        b, c, h, w = x.size()

        # 空间维度上的像素点数量 (用于计算方差)
        n = w * h - 1

        # 计算 (x - mu)^2
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)

        # 计算能量函数 (Based on neuro-science theory)
        # y 代表每个神经元的重要性权重
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        # 加权输出
        return x * self.activaton(y)

class SimAMResBlock(nn.Module):
    # SimAM Residual Bottleneck Block
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        # c1: 输入通道, c2: 输出通道 (通常 c1==c2), e: 膨胀/收缩系数
        super().__init__()
        c_hidden = int(c2 * e)  # 隐藏层通道数，比如减半

        # 1. 1x1 卷积降维
        self.cv1 = Conv(c1, c_hidden, 1, 1)

        # 2. SimAM 注意力 (无参)
        self.simam = SimAM()

        # 3. 1x1 卷积升维 (恢复通道数以进行残差连接)
        self.cv2 = Conv(c_hidden, c2, 1, 1)

        # 残差连接条件
        self.add = shortcut and c1 == c2

    def forward(self, x):
        # 流程：输入 -> 降维 -> SimAM -> 升维 -> 与原输入相加
        y = self.cv2(self.simam(self.cv1(x)))
        return x + y if self.add else y

if __name__ == '__main__':
    x = torch.randn(1, 32, 16, 16)
    model = SimAMResBlock(32, 32)
    print(model(x).shape)