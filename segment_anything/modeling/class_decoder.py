import torch
import torch.nn as nn


class ClassDecoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, out_dim)
        self.conv1 = nn.Conv2d(256, 3, kernel_size=1, stride=1, padding=0)
        self.relu = nn.LeakyReLU()
        self.norm = nn.BatchNorm1d(128)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.conv1(x)
        if len(x.shape) == 4:
            x = torch.flatten(x, start_dim=1)
        elif len(x.shape) == 3:
            raise ValueError(f'Unexpected feature size {x.shape}')
        
        x = self.fc1(x)
        x = self.norm(x)
        shortcut = x
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