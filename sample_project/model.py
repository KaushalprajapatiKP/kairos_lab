import torch
import torch.nn as nn


class SlowAttention(nn.Module):
    """Attention mechanism with slow manual implementation — bottleneck 3."""

    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Slow manual attention score computation — bottleneck
        batch_size = Q.shape[0]
        scores = []
        for i in range(batch_size):
            row_scores = []
            for j in range(batch_size):
                score = 0.0
                for k in range(self.dim):
                    score += Q[i][k].item() * K[j][k].item()
                score = score / (self.dim ** 0.5)
                row_scores.append(score)
            scores.append(row_scores)

        return torch.tensor(scores)


class SlowMLP(nn.Module):
    """MLP with slow manual activation — bottleneck 4."""

    def __init__(self, input_dim=128, hidden_dim=256, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def slow_relu(self, x):
        """Manual ReLU — should be torch.relu."""
        result = []
        for i in range(x.shape[0]):
            row = []
            for j in range(x.shape[1]):
                val = x[i][j].item()
                row.append(val if val > 0 else 0.0)
            result.append(row)
        return torch.tensor(result)

    def forward(self, x):
        x = self.fc1(x)
        x = self.slow_relu(x)
        x = self.fc2(x)
        return x