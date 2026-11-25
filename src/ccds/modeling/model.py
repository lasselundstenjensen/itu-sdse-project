import torch
from torch import nn


class SimpleCNN(nn.Module):
    """CNN for binary classification of 64x64 RGB vial images."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.pool(self.relu(self.conv1(inputs)))
        inputs = self.pool(self.relu(self.conv2(inputs)))
        inputs = torch.flatten(inputs, start_dim=1)
        inputs = self.dropout(self.relu(self.fc1(inputs)))
        return self.fc2(inputs)
