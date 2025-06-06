import pandas as pd
import torch
from torch.utils.data import Dataset

class Dataloader(Dataset):
    def __init__(self, df):
        if isinstance(df, str):  # 如果传入的是路径
            df = pd.read_csv(df)
        self.features = torch.tensor(df[[
            "log_return", "SMA_10", "price_bias", "RSI_14", 
            "MACD_diff", "bollinger_width", "volume_bias"
        ]].values, dtype=torch.float32)
        self.returns = torch.tensor(df["log_return"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.returns[idx]
