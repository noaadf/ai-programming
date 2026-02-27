"""
Task 7: Model Analysis
Functions for visualizing training results and confusion matrix
"""
import os
import sys

# Add parent directory and current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import sys

from config import SAVE_DIR, CLASS_NAMES

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history: Dictionary with 'train_loss', 'train_acc', 'val_loss', 'val_acc'
        save_path: Path to save the figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Plot accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix for model predictions.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the figure (optional)
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14)

    # Plot normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()

    plt.close()

    # Print per-class accuracy
    print("\nPer-class accuracy:")
    for i, class_name in enumerate(CLASS_NAMES):
        if cm[i].sum() > 0:
            acc = cm[i, i] / cm[i].sum() * 100
            print(f"  {class_name}: {acc:.2f}%")

    # Find most confused pairs
    print("\nMost confused class pairs:")
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    for _ in range(3):  # Top 3 confusions
        max_val = cm_no_diag.max()
        if max_val == 0:
            break
        i, j = np.unravel_index(cm_no_diag.argmax(), cm_no_diag.shape)
        print(f"  {CLASS_NAMES[i]} confused as {CLASS_NAMES[j]}: {max_val} times")
        cm_no_diag[i, j] = 0


def visualize_predictions(model, test_loader, class_names, num_images=16, save_path=None):
    """
    Visualize model predictions on sample test images.

    Args:
        model: Trained model
        test_loader: DataLoader for test data
        class_names: List of class names
        num_images: Number of images to display
        save_path: Path to save the figure (optional)
    """
    model.eval()
    images_shown = 0

    # Denormalization for visualization
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    fig = plt.figure(figsize=(12, 12))
    rows = int(np.ceil(num_images / 4))

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), targets

            # Get predictions
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            # Plot images
            for i in range(min(len(inputs), num_images - images_shown)):
                images_shown += 1

                # Denormalize image
                img = inputs[i].cpu() * std + mean
                img = torch.clamp(img, 0, 1)

                ax = fig.add_subplot(rows, 4, images_shown)
                img = img.permute(1, 2, 0).numpy()
                ax.imshow(img)
                ax.axis('off')

                # Color code correct/incorrect
                true_label = class_names[targets[i].item()]
                pred_label = class_names[predicted[i].item()]
                color = 'green' if targets[i].item() == predicted[i].item() else 'red'
                ax.set_title(f'True: {true_label}\nPred: {pred_label}', color=color, fontsize=10)

            if images_shown >= num_images:
                break

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved prediction visualization to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # Example usage
    print("This module provides visualization functions for model analysis.")
    print("Import and use these functions in your main training script.")
