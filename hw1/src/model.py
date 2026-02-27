"""
Task 2: CNN Model Definition
Implements a CNN for CIFAR-10 classification:
- 3 conv layers with ReLU and MaxPool
- 2 fully connected layers
"""
import os
import sys

# Add parent directory and current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
from config import INPUT_CHANNELS, NUM_CLASSES, CONV1_OUT, CONV2_OUT, CONV3_OUT, FC1_HIDDEN


class CNN(nn.Module):
    """
    Convolutional Neural Network for CIFAR-10 classification.

    Architecture:
        - Conv Block 1: Conv2d(3->64) + ReLU + MaxPool
        - Conv Block 2: Conv2d(64->128) + ReLU + MaxPool
        - Conv Block 3: Conv2d(128->256) + ReLU + MaxPool
        - FC Layer 1: Linear(256*4*4 -> 1024) + ReLU + Dropout
        - FC Layer 2: Linear(1024 -> 10)
    """

    def __init__(self, dropout_rate=0.5):
        """
        Initialize the CNN model.

        Args:
            dropout_rate (float): Dropout probability for FC layer
        """
        super(CNN, self).__init__()

        # First convolutional block
        # Input: 32 x 32 x 3
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, CONV1_OUT, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional block
        # Input: 16 x 16 x 64
        self.conv2 = nn.Conv2d(CONV1_OUT, CONV2_OUT, kernel_size=3, padding=1)

        # Third convolutional block
        # Input: 8 x 8 x 128
        self.conv3 = nn.Conv2d(CONV2_OUT, CONV3_OUT, kernel_size=3, padding=1)

        # Fully connected layers
        # Input: 4 x 4 x 256 = 4096
        self.fc1 = nn.Linear(CONV3_OUT * 4 * 4, FC1_HIDDEN)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(FC1_HIDDEN, NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, 10)
        """
        # Conv block 1: 32x32x3 -> 16x16x64
        x = self.pool(F.relu(self.conv1(x)))

        # Conv block 2: 16x16x64 -> 8x8x128
        x = self.pool(F.relu(self.conv2(x)))

        # Conv block 3: 8x8x128 -> 4x4x256
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten: 4x4x256 -> 4096
        x = x.view(x.size(0), -1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_num_params(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = CNN()
    print(f"Model architecture:\n{model}\n")
    print(f"Total trainable parameters: {model.get_num_params():,}")

    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
