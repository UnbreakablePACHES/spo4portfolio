import torch
import torch.nn as nn

class LinearPredictorTorch(nn.Module):
    """
    一个简单的线性预测器:y = Wx + b
    输入:
        input_dim: 特征维度
    """
    def __init__(self, input_dim):
        super(LinearPredictorTorch, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)  # 输出 shape: [batch_size]