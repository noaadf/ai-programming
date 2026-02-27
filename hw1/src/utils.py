"""
Utility functions for training
"""
import os
import sys

# Add parent directory and current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import os
import sys
from itertools import product

from config import SAVE_DIR
from train import train_model, validate
from model import CNN


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, epoch, val_acc, val_loss, filename):
    """Save model checkpoint."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
    }
    torch.save(checkpoint, os.path.join(SAVE_DIR, filename))
    print(f"Saved checkpoint: {filename}")


def load_checkpoint(model, filename, optimizer=None):
    """Load model checkpoint."""
    checkpoint_path = os.path.join(SAVE_DIR, filename)
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {filename} not found!")
        return None

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (val_acc: {checkpoint['val_acc']:.2f}%)")
    return checkpoint


def hyperparameter_search(train_loader, val_loader, param_grid):
    """
    Task 5: Perform hyperparameter tuning.

    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        param_grid: Dictionary with hyperparameters to search
            Example: {'lr': [0.01, 0.001], 'batch_size': [64, 128]}

    Returns:
        list: Results for each hyperparameter combination
    """
    from config import DEVICE

    # Get all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))

    results = []

    print(f"Starting hyperparameter search with {len(combinations)} combinations...")

    for i, combination in enumerate(combinations):
        params = dict(zip(keys, combination))
        print(f"\n[{i+1}/{len(combinations)}] Testing: {params}")

        # Create model with current params
        model = CNN(dropout_rate=params.get('dropout_rate', 0.5)).to(DEVICE)

        # Train for a few epochs (use fewer epochs for hyperparameter search)
        num_epochs = params.get('num_epochs', 10)
        lr = params.get('lr', 0.01)
        momentum = params.get('momentum', 0.9)

        history = train_model(
            model, train_loader, val_loader,
            num_epochs=num_epochs, lr=lr, momentum=momentum,
            save_best=False
        )

        # Store results
        best_val_acc = max(history['val_acc'])
        results.append({
            'params': params,
            'best_val_acc': best_val_acc,
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1],
            'history': history
        })

        print(f"  -> Best val_acc: {best_val_acc:.2f}%")

    # Sort results by validation accuracy
    results.sort(key=lambda x: x['best_val_acc'], reverse=True)

    print("\n" + "="*60)
    print("Hyperparameter Search Results:")
    print("="*60)
    for i, result in enumerate(results[:5]):  # Top 5
        print(f"\n#{i+1} - Val Acc: {result['best_val_acc']:.2f}%")
        print(f"   Params: {result['params']}")

    return results


def grid_search_results_to_table(results):
    """Convert hyperparameter search results to a formatted table."""
    print("\n" + "="*80)
    print(f"{'Rank':<5} {'Val Acc':<10} {'Train Acc':<12} {'Params'}")
    print("="*80)

    for i, result in enumerate(results):
        params_str = ", ".join(f"{k}={v}" for k, v in result['params'].items())
        print(f"{i+1:<5} {result['best_val_acc']:<10.2f} {result['final_train_acc']:<12.2f} {params_str}")
