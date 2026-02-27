"""
Main script for CNN training on CIFAR-10
Performs all tasks from the homework assignment
"""

import os
import sys
import argparse
import torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import *
from src.data import get_cifar10_dataloaders
from src.model import CNN
from src.train import train_model, test
from src.analysis import plot_training_history, plot_confusion_matrix
from src.utils import hyperparameter_search, grid_search_results_to_table

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


def main():
    parser = argparse.ArgumentParser(description='Train CNN on CIFAR-10')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'hpo', 'all'],
                        help='Mode: train, test, hyperparameter search, or all')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to saved model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory to store/load data')

    args = parser.parse_args()

    print("="*60)
    print("CNN Image Classification on CIFAR-10")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print("="*60)

    # Task 1: Load data
    print("\n[Task 1] Loading CIFAR-10 dataset...")
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir
    )
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")

    if args.mode in ['train', 'all']:
        # Task 2: Build model
        print("\n[Task 2] Building CNN model...")
        model = CNN(dropout_rate=args.dropout).to(DEVICE)
        print(f"  Model parameters: {model.get_num_params():,}")

        # Load checkpoint if specified
        start_epoch = 0
        if args.load_model:
            checkpoint = torch.load(args.load_model, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"  Loaded checkpoint from epoch {start_epoch}")

        # Task 3 & 5: Train model
        print("\n[Task 3 & 5] Training model...")
        history = train_model(
            model, train_loader, val_loader,
            num_epochs=args.epochs,
            lr=args.lr,
            momentum=args.momentum,
            save_best=True
        )

        # Task 7: Plot training history
        print("\n[Task 7] Visualizing training history...")
        plot_training_history(history, save_path=os.path.join(SAVE_DIR, 'training_history.png'))

        # Task 4 & 6: Test best model
        print("\n[Task 4 & 6] Testing best model...")
        # Load best model
        best_model_path = os.path.join(SAVE_DIR, 'best_model.pth')
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_acc, predictions, true_labels = test(model, test_loader)
        print(f"  Test Accuracy: {test_acc:.2f}%")

        # Task 7: Confusion matrix
        print("\n[Task 7] Generating confusion matrix...")
        plot_confusion_matrix(true_labels, predictions, save_path=os.path.join(SAVE_DIR, 'confusion_matrix.png'))

    if args.mode == 'test':
        if not args.load_model:
            print("Error: Please specify --load-model for testing mode")
            return

        print("\n[Task 4] Loading model and testing...")
        model = CNN(dropout_rate=args.dropout).to(DEVICE)
        checkpoint = torch.load(args.load_model, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_acc, predictions, true_labels = test(model, test_loader)
        print(f"  Test Accuracy: {test_acc:.2f}%")

        plot_confusion_matrix(true_labels, predictions, save_path=os.path.join(SAVE_DIR, 'confusion_matrix.png'))

    if args.mode == 'hpo':
        # Task 5: Hyperparameter optimization
        print("\n[Task 5] Starting hyperparameter optimization...")

        param_grid = {
            'lr': [0.1, 0.01, 0.001],
            'batch_size': [64, 128, 256],
            'momentum': [0.9, 0.95],
            'num_epochs': [10]  # Use fewer epochs for HPO
        }

        # Override with command line if needed
        results = hyperparameter_search(train_loader, val_loader, param_grid)
        grid_search_results_to_table(results)

        # Save best hyperparameters
        best_params = results[0]['params']
        with open(os.path.join(SAVE_DIR, 'best_hparams.txt'), 'w') as f:
            f.write(f"Best hyperparameters:\n")
            for k, v in best_params.items():
                f.write(f"  {k}: {v}\n")
            f.write(f"\nBest validation accuracy: {results[0]['best_val_acc']:.2f}%\n")
        print(f"\nSaved best hyperparameters to {os.path.join(SAVE_DIR, 'best_hparams.txt')}")

    print("\n" + "="*60)
    print("All tasks completed!")
    print("="*60)


if __name__ == "__main__":
    main()
