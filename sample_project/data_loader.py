import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SlowDataset(Dataset):
    """Dataset with intentionally slow preprocessing."""

    def __init__(self, size=1000, feature_dim=128):
        self.size = size
        self.feature_dim = feature_dim
        self.data = self._generate_data()

    def _generate_data(self):
        """Slow element-wise data generation — bottleneck 1."""
        result = []
        for i in range(self.size):
            row = []
            for j in range(self.feature_dim):
                val = float(i * j) / (self.size * self.feature_dim)
                row.append(val)
            result.append(row)
        return result

    def normalize(self, data):
        """Slow manual normalization — bottleneck 2."""
        means = []
        for col in range(self.feature_dim):
            col_sum = 0
            for row in range(len(data)):
                col_sum += data[row][col]
            means.append(col_sum / len(data))

        normalized = []
        for row in data:
            norm_row = []
            for j, val in enumerate(row):
                norm_row.append(val - means[j])
            normalized.append(norm_row)
        return normalized

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        features = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(idx % 10, dtype=torch.long)
        return features, label


def get_dataloader(size=1000, feature_dim=128, batch_size=32):
    dataset = SlowDataset(size, feature_dim)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)