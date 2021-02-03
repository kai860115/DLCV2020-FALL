from torch import nn
import torch


class Convnet(nn.Module):
    def __init__(self, in_channels=3, hid_channels=64, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, out_channels)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class Hallucinator(nn.Module):
    def __init__(self, z_dim=1600):
        super().__init__()
        self.h = nn.Sequential(
            nn.Linear(1600 + z_dim, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            nn.Linear(800, 1600),
            nn.ReLU()
        )

    def forward(self, x, z):
        x = torch.cat((x, z), -1)
        x = self.h(x)
        return x

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1600, 800),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(800, 400)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x