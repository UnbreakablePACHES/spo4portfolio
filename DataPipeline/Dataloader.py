import torch
from torch.utils.data import Dataset

class Dataloader(Dataset):
    def __init__(self, features_df, labels_df, num_assets):
        """
        参数:
        - features_df: shape = (num_samples, num_assets * num_features)
        - labels_df: shape = (num_samples, num_assets)
        - num_assets: ETF 数量（如 8）
        """
        num_features = features_df.shape[1] // num_assets

        self.X = features_df.values.reshape(-1, num_assets, num_features)  # shape: (N, 8, 7)
        self.y = labels_df.values  # shape: (N, 8)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)  # shape: (8, 7)
        y = torch.tensor(self.y[idx], dtype=torch.float32)  # shape: (8,)
        return x, y
