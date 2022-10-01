from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP5HiddenLayers(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 2048)
        self.fc4 = nn.Linear(2048, 512)
        self.fc5 = nn.Linear(512, 128)
        self.fc6 = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x


class CNN5HiddenLayers(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 128, 3)
        self.conv3 = nn.Conv2d(128, 512, 3)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 128)
        self.fc6 = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.conv3(x)
        x = F.avg_pool2d(x, kernel_size=(x.shape[2], x.shape[3]))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x


models = {
    "CNN5HiddenLayers": CNN5HiddenLayers,
    "MLP5HiddenLayers": MLP5HiddenLayers
}
