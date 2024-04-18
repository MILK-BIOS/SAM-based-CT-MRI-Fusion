import torch
import torch.nn as nn


class ClassDecoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, in_dim)
        self.fc4 = nn.Linear(in_dim, out_dim)
        self.relu = nn.GELU()
        self.norm = nn.BatchNorm1d()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.fc1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = x + shortcut
        shortcut = x
        x = self.norm(x)
        x = self.relu(x)
        x = x + shortcut
        x = self.norm(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x