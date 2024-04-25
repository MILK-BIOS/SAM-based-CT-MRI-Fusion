import torch
import torch.nn as nn


class SiameseCnnDecoder(nn.Module):
    def __init__(self, in_channels: int = 256, out_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 3, 3, padding=1)
        self.conv4 = nn.Conv2d(3, 1, 1, padding=0)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x