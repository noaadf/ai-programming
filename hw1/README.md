# CNN Image Classification on CIFAR-10

Deep Learning Homework 1 - CNN with adaptive loss and sampling for handling class imbalance.

## Project Structure

```
cnn-hw1/
├── data/                             # Dataset cache (auto-downloaded)
├── models/                           # Saved models, plots, and logs
├── src/                              # Source code
│   ├── config.py                     # Configuration (hyperparameters, paths)
│   ├── data.py                       # Data loading and augmentation
│   ├── model.py                      # CNN model definition
│   ├── train.py                      # Training and validation functions
│   ├── analysis.py                   # Visualization (confusion matrix, plots)
│   └── utils.py                      # Hyperparameter search utilities
├── main.py                           # Basic training script
├── cat_dog_experiment_adaptive.py    # Cat-Dog confusion experiment with adaptive methods
└── README.md                         # This file
```

## Environment Setup

### Remote Server (H100 GPU)

```bash
# SSH to server
ssh H100_l

# Activate conda environment
source /home/zhicheng/gyh/miniconda3/etc/profile.d/conda.sh
conda activate cnn-hw1

# Navigate to project
cd /home/zhicheng/gyh/ai-pg/cnn-hw1
```

## Cat-Dog Confusion Problem

CIFAR-10 contains images of cats and dogs that are frequently confused by the model. This project implements **Dynamic Focal Loss** and **Focal-Guided Adaptive Sampling** to address this issue.

### Problem Analysis

- Cats (class 3) and dogs (class 5) are visually similar
- Standard CrossEntropy treats all samples equally
- Easy samples dominate the gradient, hard samples get insufficient attention

### Solution: Dynamic Focal Loss

```python
# Standard Focal Loss
FL(p_t) = -(1 - p_t)^γ × log(p_t)

# Dynamic scale adjustment (every 5 epochs)
scale = 1 / E[(1-p_t)^γ]
```

**Key innovation**: The scale factor is dynamically adjusted based on current model predictions, ensuring total weight equals sample count.

| Epoch | mean_weight | scale |
|-------|-------------|-------|
| 5     | 0.300       | 3.33  |
| 10    | 0.233       | 4.30  |
| 20    | 0.189       | 5.28  |
| 30    | 0.110       | 9.09  |
| 50    | 0.071       | 14.10 |

### Solution: Focal-Guided Adaptive Sampling

Samples are weighted by `(1 - p_t)^γ` during data loading:
- **Hard samples** (low p_t) → higher sampling probability
- **Easy samples** (high p_t) → lower sampling probability

Progression during training:
```
Epoch 4:  easy=9.2%,  hard=56.2%
Epoch 20: easy=50.6%, hard=20.9%
Epoch 50: easy=70.0%, hard=9.0%
```

## Experimental Results

### Final Comparison (50 epochs, 4x H100 GPU)

| Configuration | Val Acc | Confusion Rate | vs CE |
|---------------|---------|----------------|-------|
| CrossEntropy (baseline) | 85.96% | 12.02% | - |
| **DynamicFocalLoss** | **86.78%** | **10.78%** | +0.82% / +1.24% |
| CE + FocalSampler | 86.78% | 11.25% | +0.82% / +0.77% |
| DFL + Sampler | 86.06% | 13.05% | +0.10% / -1.03% |

**Conclusion**: DynamicFocalLoss alone achieves the best results—both highest accuracy and lowest cat-dog confusion rate.

### Key Findings

1. **Dynamic scale is critical**: Removing the upper limit on scale (from capped at 10 to reaching 14.1) improved performance
2. **Double weighting hurts**: Combining DFL + Sampler creates conflicting gradients
3. **Adaptivity matters**: Both dynamic FL and adaptive sampler independently improve over static CE

## Running the Experiment

```bash
# Run on 4 GPUs [4,5,6,7]
python cat_dog_experiment_adaptive.py --epochs 50

# Monitor progress
tail -f adaptive_train.log

# Check GPU usage
nvidia-smi
```

## Model Architecture

```
Input: 32x32x3 (RGB image)

CNN:
  conv1: Conv2d(3 -> 64, 3x3) + ReLU + MaxPool2d(2x2)  -> 16x16x64
  conv2: Conv2d(64 -> 128, 3x3) + ReLU + MaxPool2d(2x2) -> 8x8x128
  conv3: Conv2d(128 -> 256, 3x3) + ReLU + MaxPool2d(2x2) -> 4x4x256
  fc1:   Linear(4096 -> 1024) + ReLU + Dropout(0.5)
  fc2:   Linear(1024 -> 10)

Total Parameters: 4,576,394
```

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | SGD (lr=0.01, momentum=0.9) |
| Weight Decay | 5e-4 |
| LR Schedule | MultiStepLR (milestones=[25, 40], gamma=0.1) |
| Batch Size | 128 |
| Dropout | 0.5 |
| Data Augmentation | RandomCrop(32, padding=4), RandomHorizontalFlip |
| GPUs | [4, 5, 6, 7] (DataParallel) |

## Output Files

| File | Description |
|------|-------------|
| `models/cat_dog_experiment_v4.png` | Final comparison plot |
| `models/adaptive_train.log` | Training log with epoch-by-epoch metrics |

## Tasks Checklist

- [x] Task 1: Data preparation with augmentation
- [x] Task 2: Build CNN model
- [x] Task 3: Training loop
- [x] Task 4: Test/validation function
- [x] Task 5: Hyperparameter tuning
- [x] Task 6: Full training with best model saving
- [x] Task 7: Confusion matrix visualization
- [x] **Dynamic Focal Loss implementation**
- [x] **Focal-Guided Adaptive Sampling**
