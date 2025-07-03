# # Experimental setup

import pandas as pd
import numpy as np
import torch
import random

from models.LinearInferencer import LinearPredictorTorch
from models.PortfolioModel import PortfolioModel # 定义投资组合优化模型
from DataPipeline.Dataloader import PortfolioDataset # 将数据读取为tensor
from torch.utils.data import DataLoader
from DataPipeline.DataBuilder import build_dataset

from pyepo.func.surrogate import SPOPlus

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

## Train
features_df, labels_df = build_dataset(tickers) # 读取输入特征和输出特征
num_assets = len(tickers) # 维度=tickers数组长度
dataset = PortfolioDataset(features_df, labels_df, num_assets=num_assets) # 构建 PyTorch Dataset

batch_size = 32
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
for x_batch, y_batch in train_loader:
    print(x_batch.shape, y_batch.shape)
    break

predictor = LinearPredictorTorch(input_dim=7 * 8, num_assets=8)
x_batch, _ = next(iter(train_loader))
c_hat = predictor(x_batch)  # 输出 shape: [batch_size, 8]
c_hat.shape

predictor.train()
total_loss = 0.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for x_batch, c_true_batch in train_loader:
    x_batch = x_batch.to(device)         # [batch_size, 8, 7]
    c_true_batch = c_true_batch.to(device)  # [batch_size, 8]

x_batch, c_true_batch = next(iter(train_loader))
x_batch = x_batch.to(device)         # shape: [batch_size, 8, 7]
c_true_batch = c_true_batch.to(device)  # shape: [batch_size, 8]

optmodel = PortfolioModel(n_assets=8, budget=1.0) # 实例化优化模型
optimizer = torch.optim.Adam(predictor.parameters(), lr=0.001) # 实例化优化器
spo_loss_fn = SPOPlus(optmodel, processes=1, solve_ratio=1.0, reduction="mean") # 实例化spoloss

print(type(spo_loss_fn))
import inspect
print(inspect.getfile(spo_loss_fn.__class__))

total_loss = 0.0  # 初始化累计损失

## Train main script
for i in range(x_batch.size(0)):  # 遍历这个 batch 中的每个样本
    x_sample = x_batch[i].unsqueeze(0).to(device)   # [1, 8, 7]
    c_true = c_true_batch[i].to(device)             # [8]

    optimizer.zero_grad()

    # 前向传播：预测 \hat{c}
    c_hat = predictor(x_sample).squeeze(0)          # [8]

    # 设置目标向量并求解 z*(c_true)
    optmodel.setObj(c_true.detach().cpu().numpy())  # ✅ 设置目标函数
    z_star_np, obj_val = optmodel.solve()           # ✅ 无参数调用

    # 转为 PyTorch Tensor
    z_star = torch.tensor(z_star_np, dtype=torch.float32, device=device)  # [8]
    true_obj = torch.tensor(obj_val, dtype=torch.float32, device=device)  # []

    # 添加 batch 维度，确保 shape = [1, 8] / [1]
    c_hat = c_hat.unsqueeze(0)
    c_true = c_true.unsqueeze(0)
    z_star = z_star.unsqueeze(0)
    true_obj = true_obj.unsqueeze(0)

    # 计算 SPO+ loss
    loss = spo_loss_fn(c_hat, c_true, z_star, true_obj)

    # 反向传播 + 参数更新
    loss.backward()
    optimizer.step()

    total_loss += loss.item()  # 先验证一个样本是否成功
print(f"z_star: {z_star}")
print(f"true_obj: {true_obj.item():.4f}")
print(f"loss: {loss.item():.4f}")