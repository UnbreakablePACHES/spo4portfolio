import torch
from torch.utils.data import Dataset

class PortfolioDataset(Dataset):
    def __init__(self, features_df, labels_df, num_assets):
        """
        parameter:
        - features_df: shape = (num_samples, num_assets * num_features)
        - labels_df: shape = (num_samples, num_assets)
        - num_assets: Number of assets
        """
        num_features = features_df.shape[1] // num_assets

        self.X = features_df.values.reshape(-1, num_assets, num_features) 
        # shape: (T = Time steps, N = Number of assets, F = Number of features)
        self.y = labels_df.values  # shape: (T, N)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)  # shape: (T, F)
        y = torch.tensor(self.y[idx], dtype=torch.float32)  # shape: (F,)
        return x, y
