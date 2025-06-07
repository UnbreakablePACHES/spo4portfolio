import torch
import torch.nn as nn

class LinearPredictorTorch(nn.Module):
    def __init__(self, input_dim, num_assets):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_assets)  # 一次性预测全部 ETF

    def forward(self, x):
        """
        x: shape = (batch_size, num_assets, input_dim)
        输出: shape = (batch_size, num_assets)
        """
        x = x.view(x.size(0), -1)  # (batch, 8, 7) → (batch, 56)
        return self.linear(x)
