"""
Configuration file for CNN training on CIFAR-10
"""

import torch

# Data settings
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)  # CIFAR-10 dataset mean
CIFAR10_STD = (0.2470, 0.2435, 0.2616)   # CIFAR-10 dataset std

# Data augmentation settings
RANDOM_CROP_SIZE = 32         # Size for random crop (after padding)
RANDOM_CROP_PADDING = 4       # Padding for random crop

# Model architecture
INPUT_CHANNELS = 3            # RGB images
NUM_CLASSES = 10              # CIFAR-10 has 10 classes

CONV1_OUT = 64
CONV2_OUT = 128
CONV3_OUT = 256

FC1_HIDDEN = 1024             # Hidden layer size

# Training hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 20

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
DATA_ROOT = './data'
MODEL_SAVE_PATH = './models/best_model.pth'
CHECKPOINT_PATH = './models/checkpoint.pth'
SAVE_DIR = './models'

# CIFAR-10 class names for visualization
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
