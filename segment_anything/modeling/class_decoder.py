import torch
import torch.nn as nn


class ClassDecoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.relu = nn.LeakyReLU()
        self.norm= nn.BatchNorm2d(16)
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.softmax = nn.Softmax(dim=0)
        self.global_avg_pool = GlobalAveragePooling()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        feature_map = x
        x = self.global_avg_pool(x)
        x = self.fc(x)
        x = self.softmax(x)

        cam = torch.einsum('ij,bjhw->bihw', self.fc.weight, feature_map)
        
        return x, cam
    
class GlobalAveragePooling(nn.Module):
    def forward(self, x):
        return torch.mean(x, dim=(2, 3))  # 对 height 和 width 维度求平均