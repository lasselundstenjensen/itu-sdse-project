from torch import nn


class SimpleCNN(nn.Module):
    """
    Simple CNN for binary classification of glass vial images.

    Input: 64x64 RGB images (3 channels) - resized from native 1600x768
    Output: Single logit (positive = defective, negative = clean)
    """

    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers. After 2 max pooling layers: 64x64 -> 32x32 -> 16x16
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)

        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Convolutional layers with ReLU and max pooling
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        # Flatten for fully connected layers
        x = x.view(-1, 64 * 16 * 16)

        # No sigmoid here: the model is trained with BCEWithLogitsLoss,
        # callers apply torch.sigmoid for probabilities.
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
