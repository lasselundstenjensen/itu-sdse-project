"""CNN architecture for binary vial classification."""

import torch
import torch.nn as nn

ARCHITECTURE_NAME = "SimpleCNN"


class SimpleCNN(nn.Module):
    """Small CNN for glass-vial images.

    Input:  ``(N, 3, 64, 64)`` normalised RGB tensors.
    Output: ``(N, 1)`` raw logits. Positive means defective. Callers apply
            ``torch.sigmoid`` to get probabilities; the model is trained with
            ``BCEWithLogitsLoss``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 64x64 -> 32x32 -> 16x16 after two poolings
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
