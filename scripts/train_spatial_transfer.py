#!/usr/bin/env python3
"""
Spatial transfer experiments: train a PointNet model on one region (core/periphery)
and evaluate on the other, with optional semi‑supervised setting.
"""

import argparse
import logging
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Project modules (adjust import if your package name differs)
from pcdg.pointnet import get_model
from pcdg.pcdata import PcData, PcDataset
from pcdg.augment import augment_point_cloud
from pcdg.utils import (
    split_data,                # your existing function for core/periphery splits
    set_seed,
    get_device,
    train_model,                # we'll define this in utils (or keep local)
    setup_logger,
)


# -----------------------------------------------------------------------------
# Training function (if not moved to utils)
# -----------------------------------------------------------------------------
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    device: torch.device,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    save_model_title: str = "model",
    results_dir: str = "./results/losses",
) -> str:
    """
    Train a model, save the best checkpoint based on validation loss,
    and log losses/accuracies.

    Args:
        model: PyTorch model.
        train_loader: DataLoader for training.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        num_epochs: Number of epochs.
        device: Device to train on.
        val_loader: DataLoader for validation.
        test_loader: Optional DataLoader for test set (only for logging).
        save_model_title: Base name for saved model and loss file.
        results_dir: Directory to save loss CSV.

    Returns:
        Path to the saved best model checkpoint.
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("./savedmodel", exist_ok=True)
    model_path = os.path.join("./savedmodel", f"{save_model_title}.pth")
    loss_path = os.path.join(results_dir, f"loss_{save_model_title}.csv")

    best_val_loss = float("inf")
    history = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for points, targets in train_loader:
            points, targets = points.to(device), targets.to(device)
            # targets shape: (batch,) after collation, ensure long
            targets = targets.long()
            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for points, targets in val_loader:
                points, targets = points.to(device), targets.to(device).long()
                outputs = model(points)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        # Test (optional, only for logging)
        test_acc = None
        if test_loader is not None:
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for points, targets in test_loader:
                    points, targets = points.to(device), targets.to(device).long()
                    outputs = model(points)
                    preds = outputs.argmax(dim=1)
                    test_correct += (preds == targets).sum().item()
                    test_total += targets.size(0)
            test_acc = test_correct / test_total if test_total > 0 else 0.0

        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_val_loss)

        # Logging
        logging.info(
            f"Epoch {epoch+1:2d}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.3f} | "
            f"Test Acc: {test_acc:.3f}" + (f" | LR: {current_lr:.2e}")
        )
        history.append([epoch+1, avg_train_loss, avg_val_loss, test_acc, current_lr])

    # Save loss history
    pd.DataFrame(history, columns=["epoch", "train_loss", "val_loss", "test_acc", "lr"]).to_csv(loss_path, index=False)
    return model_path


# -----------------------------------------------------------------------------
# Main experiment function
# -----------------------------------------------------------------------------
def run_spatial_transfer(
    veg: str,
    channel: str,
    split_method: str,
    augmentation: bool,
    semi_supervised: bool,
    model_type: str,
    num_epochs: int,
    device: torch.device,
    data_root: str = "./data",
    results_root: str = "./results",
    seed: int = 42,
) -> None:
    """
    Run a spatial transfer experiment: train on one region (core/periphery)
    and evaluate on the other, with optional semi‑supervised learning.

    Args:
        veg: Vegetable name ('mushroom', 'broccoli', 'combine').
        channel: Channel name ('a', 'b', 'c').
        split_method: How to split data ('core_periphery', 'periphery_core', 'random').
        augmentation: Whether to apply data augmentation.
        semi_supervised: If True, include a few samples from the test domain in training.
        model_type: Model variant ('geo' or 'geo_int').
        num_epochs: Number of training epochs.
        device: Torch device.
        data_root: Root directory containing 'cloud/{veg}' and ground truth CSV.
        results_root: Root directory for saving outputs.
        seed: Random seed for reproducibility.
    """
    set_seed(seed)

    # Create a descriptive title for saving outputs
    save_title = f"{split_method}_{veg}_{channel}_aug_{augmentation}_semi_{semi_supervised}_{model_type}"
    setup_logger(os.path.join(results_root, "losses", f"{save_title}.log"))

    logging.info(f"Experiment: {save_title}")
    logging.info(f"Settings: veg={veg}, channel={channel}, split={split_method}, aug={augmentation}, semi={semi_supervised}")

    # Load data
    folder_name = os.path.join(data_root, "cloud", veg)
    ground_truth_file = os.path.join(data_root, "mc.csv")
    pcd = PcData(folder_name, ground_truth_file, channel_name=channel)

    # Split data according to method (core/periphery, etc.)
    # split_data returns (X_train, y_train, X_val, y_val, X_test, y_test)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        pcd, method=split_method, semi_supervised=semi_supervised
    )

    # Data augmentation (if enabled)
    if augmentation:
        X_train, y_train = augment_point_cloud(X_train, y_train)

    # Transpose for PointNet: (batch, channels, points)
    X_train = np.transpose(X_train, (0, 2, 1)).astype(np.float32)
    X_val   = np.transpose(X_val,   (0, 2, 1)).astype(np.float32)
    X_test  = np.transpose(X_test,  (0, 2, 1)).astype(np.float32)

    # Create datasets and loaders
    train_dataset = PcDataset(X_train, y_train)
    val_dataset   = PcDataset(X_val,   y_val)
    test_dataset  = PcDataset(X_test,  y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

    logging.info(f"Data sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Initialize model, loss, optimizer, scheduler
    model = get_model(model_type, dropout=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Train
    _ = train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=device,
        val_loader=val_loader,
        test_loader=test_loader,
        save_model_title=save_title,
        results_dir=os.path.join(results_root, "losses"),
    )

    logging.info("Training completed.")


# -----------------------------------------------------------------------------
# Command line interface
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spatial transfer experiments for point cloud moisture classification")
    parser.add_argument("--veg", type=str, required=True, choices=["mushroom", "broccoli", "combine"],
                        help="Vegetable dataset")
    parser.add_argument("--channel", type=str, required=True, choices=["a", "b", "c"],
                        help="Channel name (CSV file name inside each sample folder)")
    parser.add_argument("--split_method", type=str, required=True,
                        choices=["core_periphery", "periphery_core", "random"],
                        help="How to split data: train on core, test on periphery, etc.")
    parser.add_argument("--aug", action="store_true", help="Apply data augmentation")
    parser.add_argument("--semi_supervised", action="store_true",
                        help="Include a few labeled samples from test domain in training")
    parser.add_argument("--model", type=str, default="geo_int", choices=["geo", "geo_int"],
                        help="Model type: 'geo' (only geometry) or 'geo_int' (geometry + intensity)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--data_root", type=str, default="../data", help="Root data directory")
    parser.add_argument("--results_root", type=str, default="./results", help="Root results directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(cuda_id=args.cuda)

    run_spatial_transfer(
        veg=args.veg,
        channel=args.channel,
        split_method=args.split_method,
        augmentation=args.aug,
        semi_supervised=args.semi_supervised,
        model_type=args.model,
        num_epochs=args.epochs,
        device=device,
        data_root=args.data_root,
        results_root=args.results_root,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
