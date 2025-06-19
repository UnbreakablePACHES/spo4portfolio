import pandas as pd
import numpy as np
import torch
import random

from models.LinearInferencer import LinearPredictorTorch
from DataPipeline.Dataloader import PortfolioDataset
from torch.utils.data import DataLoader
from DataPipeline.DataBuilder import build_dataset
from torch import nn
from torch.optim import Adam
pd.options.display.float_format = '{:.6f}'.format
np.set_printoptions(precision=6, suppress=True)

tickers = ["EEM","EFA","JPXN","SPY","XLK",'VTI','AGG','DBC']

seed = 123

# 设置 Python 内建随机模块
random.seed(seed)

# 设置 NumPy 随机种子
np.random.seed(seed)

# 设置 PyTorch 的随机种子
torch.manual_seed(seed)

class FNNSoftmaxAllocator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_assets):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_assets)
        )

    def forward(self, preds):  # preds: (B, N)
        scores = self.net(preds)     # (B, N)
        weights = torch.softmax(scores, dim=1)
        return weights
    
def spo_plus_loss(pred_y, true_y, allocator):
    pred_weights = allocator(pred_y)        # ŵ = g(ŷ)
    oracle_weights = allocator(true_y)      # w* ≈ g(y)

    regret = torch.sum((oracle_weights - pred_weights) * true_y, dim=1)
    return regret.mean()

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


# 模型超参数
input_dim = 7         # 每个资产的特征数
num_assets = 8        # ETF 数量
hidden_dim = 32       # allocator 隐层宽度
epochs = 50           # 训练轮数
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 模型实例化
predictor = LinearPredictorTorch(input_dim * num_assets, num_assets).to(device)
allocator = FNNSoftmaxAllocator(num_assets, hidden_dim, num_assets).to(device)

# 优化器（联合训练 predictor 和 allocator）
optimizer = Adam(list(predictor.parameters()) + list(allocator.parameters()), lr=1e-3)
features_df, labels_df = build_dataset(
    tickers=["SPY", "VTI", "EFA", "EEM", "XLK", "JPXN", "AGG", "DBC"],
    start_date="2023-01-01",
    end_date="2023-12-31")

oracle_df = pd.read_csv("data/DailyOracle/oracle_weights.csv", index_col=0)
if len(features_df) != len(oracle_df):
    raise ValueError("features_df 和 oracle_df 行数不一致，不能对齐！")

labels_df = oracle_df.copy()
print(labels_df)
# 创建 dataset
dataset = PortfolioDataset(features_df, labels_df, num_assets=8)

# 创建 dataloader
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
# 训练循环
for epoch in range(epochs):
    total_loss = 0
    for x, y in train_loader:
        x = x.to(device)  # shape: (B, N, F)
        y = y.to(device)  # shape: (B, N)
        # 前向传播
        pred_y = predictor(x)  # shape: (B, N)
        loss = spo_plus_loss(pred_y, y, allocator)
    
        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1:02d} - Loss: {avg_loss:.6f}")





