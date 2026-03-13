#!/usr/bin/env python3
"""
Nested cross‑validation for PointNet‑based moisture content classification.

This script performs nested cross‑validation to tune hyperparameters (learning rate,
dropout, weight decay) and evaluate model performance on point cloud data from
different vegetable samples.

Example usage:
    python cross_val.py --veg mushroom --channel a --aug --cuda 0 --model geo_int
"""

import argparse
import json
import os
import shutil
from itertools import product
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Import project modules (adjust if your package name differs)
from pcdg.pointnet import get_model
from pcdg.pcdata import PcData, PcDataset   # using new name PcDataset
from pcdg.augment import augment_point_cloud
from pcdg.utils import (
    get_core_periphery,
    create_balanced_dataset,
    safe_collate,
    get_device,
    train_one_fold,          # we'll define this in utils or keep it here
)

# -----------------------------------------------------------------------------
# Training function (if not moved to utils)
# -----------------------------------------------------------------------------
def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    device: torch.device,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    save_model_title: str = "",
    results_dir: str = "./results/losses",
) -> str:
    """
    Train a model for a single fold, save the best checkpoint, and log losses.

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
    model_path = os.path.join("./savedmodel", f"model_{save_model_title}.pth")
    loss_path = os.path.join(results_dir, f"loss_{save_model_title}.csv")

    best_val_loss = float("inf")
    history = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for points, targets in train_loader:
            points, targets = points.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(points)
            targets = targets.squeeze(1).long() if targets.dim() > 1 else targets.long()
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
                points, targets = points.to(device), targets.to(device)
                outputs = model(points)
                targets = targets.squeeze(1).long() if targets.dim() > 1 else targets.long()
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
                    points, targets = points.to(device), targets.to(device)
                    outputs = model(points)
                    targets = targets.squeeze(1).long() if targets.dim() > 1 else targets.long()
                    preds = outputs.argmax(dim=1)
                    test_correct += (preds == targets).sum().item()
                    test_total += targets.size(0)
            test_acc = test_correct / test_total if test_total > 0 else 0.0

        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:2d}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.3f} | "
            f"Test Acc: {test_acc:.3f}" if test_acc is not None else ""
            f"LR: {current_lr:.2e}"
        )

        history.append([epoch+1, avg_train_loss, avg_val_loss, test_acc, current_lr])
        scheduler.step(avg_val_loss)

    # Save loss history
    pd.DataFrame(history, columns=["epoch", "train_loss", "val_loss", "test_acc", "lr"]).to_csv(loss_path, index=False)
    return model_path


# -----------------------------------------------------------------------------
# Nested cross‑validation
# -----------------------------------------------------------------------------
def nested_cross_validation(
    veg: str,
    channel: str,
    n_splits: int = 10,
    augmentation: bool = False,
    model_type: str = "geo_int",
    param_grid: Optional[Dict[str, List[Any]]] = None,
    fixed_params: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    data_root: str = "./data",
    results_root: str = "./results",
    num_epochs: int = 10,
) -> None:
    """
    Run nested cross‑validation with inner loop for hyperparameter tuning.

    Args:
        veg: Vegetable name ('mushroom', 'broccoli', 'combine').
        channel: Channel name ('a', 'b', 'c').
        n_splits: Number of outer folds.
        augmentation: Whether to apply data augmentation.
        model_type: Model variant ('geo' or 'geo_int').
        param_grid: Dictionary of hyperparameters to search.
        fixed_params: If provided, use these parameters directly (skip inner search).
        device: Torch device.
        data_root: Root directory containing 'cloud/{veg}' and ground truth CSV.
        results_root: Root directory for saving outputs.
        num_epochs: Number of training epochs per configuration.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Default hyperparameter grid
    if param_grid is None:
        param_grid = {
            "lr": [1e-3, 5e-4],
            "dropout": [0.2, 0.3, 0.5],
            "weight_decay": [0, 1e-3, 1e-5],
        }

    # Create parameter combinations
    if fixed_params:
        param_combinations = [(
            fixed_params["lr"],
            fixed_params["dropout"],
            fixed_params["weight_decay"],
        )]
    else:
        param_combinations = list(product(*param_grid.values()))

    # Load data
    folder_name = os.path.join(data_root, "cloud", veg)
    ground_truth_file = os.path.join(data_root, "mc.csv")
    pcd = PcData(folder_name, ground_truth_file, channel_name=channel)

    # Prepare data and labels (only core segments for this experiment)
    data = np.array([seg for seg, _ in pcd.segmented_core])
    labels = np.array([label for _, label in pcd.segmented_core])

    # Outer CV
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_folds_results = []

    # Create output directories
    fold_out_dir = os.path.join(results_root, "folds", veg)
    os.makedirs(fold_out_dir, exist_ok=True)

    for fold, (train_val_idx, test_idx) in enumerate(outer_cv.split(data, labels)):
        print(f"\n{'='*40}\nFold {fold+1}/{n_splits}\n{'='*40}")

        # Split into train+val and test
        X_train_val, y_train_val = get_core_periphery(pcd, train_val_idx)  # from utils
        X_test, y_test = get_core_periphery(pcd, test_idx)

        # Save test indices for reproducibility
        with open(os.path.join(fold_out_dir, f"test_idx_{veg}_{channel}_aug_{augmentation}_fold_{fold}.json"), "w") as f:
            json.dump(test_idx.tolist(), f)

        # Inner split (train / validation)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=fold)
        inner_train_idx, inner_val_idx = next(sss.split(X_train_val, y_train_val))
        X_train, y_train = X_train_val[inner_train_idx], y_train_val[inner_train_idx]
        X_val, y_val = X_train_val[inner_val_idx], y_train_val[inner_val_idx]

        # Data augmentation (if enabled)
        if augmentation:
            X_train, y_train = augment_point_cloud(X_train, y_train)

        # Transpose for PointNet (batch, channels, points)
        X_train = np.transpose(X_train, (0, 2, 1)).astype(np.float32)
        X_val   = np.transpose(X_val, (0, 2, 1)).astype(np.float32)
        X_test  = np.transpose(X_test, (0, 2, 1)).astype(np.float32)

        # Balance classes (optional, uses function from utils)
        X_train, y_train = create_balanced_dataset(X_train, y_train)
        X_val, y_val = create_balanced_dataset(X_val, y_val)

        # Create DataLoaders
        train_loader = DataLoader(
            PcDataset(X_train, y_train), batch_size=32, shuffle=True, collate_fn=safe_collate
        )
        val_loader = DataLoader(PcDataset(X_val, y_val), batch_size=32, shuffle=False)
        test_loader = DataLoader(PcDataset(X_test, y_test), batch_size=32, shuffle=False)

        # Inner loop: hyperparameter search
        best_val_loss = float("inf")
        best_model_path = best_loss_path = best_config = None
        best_model_title = ""

        for i, (lr, dropout, weight_decay) in enumerate(param_combinations):
            print(f"\nConfig {i+1}: LR={lr:.0e}, Dropout={dropout}, WD={weight_decay}")
            model = get_model(model_type=model_type, dropout=dropout).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
            criterion = nn.CrossEntropyLoss()

            model_title = f"fold{fold}_cfg{i}_{veg}_{channel}_aug{augmentation}_{model_type}"
            model_path = train_fold(
                model,
                train_loader,
                criterion,
                optimizer,
                scheduler,
                num_epochs,
                device,
                val_loader,
                test_loader,
                save_model_title=model_title,
                results_dir=os.path.join(results_root, "losses"),
            )

            # Read back validation loss (min over epochs)
            loss_df = pd.read_csv(os.path.join(results_root, "losses", f"loss_{model_title}.csv"))
            val_loss_min = loss_df["val_loss"].min()

            if val_loss_min < best_val_loss:
                best_val_loss = val_loss_min
                best_model_path = model_path
                best_loss_path = os.path.join(results_root, "losses", f"loss_{model_title}.csv")
                best_model_title = model_title
                best_config = {"lr": lr, "dropout": dropout, "weight_decay": weight_decay}

        # After inner search, copy best model and loss to fold directory
        shutil.copy(best_model_path, os.path.join(fold_out_dir, f"best_model_{veg}_{channel}_aug{augmentation}_fold{fold}.pth"))
        shutil.copy(best_loss_path, os.path.join(fold_out_dir, f"loss_{veg}_{channel}_aug{augmentation}_fold{fold}.csv"))
        with open(os.path.join(fold_out_dir, f"best_config_{veg}_{channel}_aug{augmentation}_fold{fold}.json"), "w") as f:
            json.dump(best_config, f, indent=2)

        # Evaluate best model on test set
        best_model = get_model(model_type=model_type, dropout=best_config["dropout"]).to(device)
        best_model.load_state_dict(torch.load(best_model_path))
        best_model.eval()

        all_preds, all_true = [], []
        with torch.no_grad():
            for points, labels in test_loader:
                points, labels = points.to(device), labels.to(device)
                outputs = best_model(points)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_true.extend(labels.cpu().numpy().tolist())

        fold_result = {"fold": fold, "true": all_true, "pred": all_preds}
        all_folds_results.append(fold_result)

    # Save overall test results
    overall_file = os.path.join(fold_out_dir, f"test_results_{veg}_{channel}_aug{augmentation}_{model_type}.json")
    with open(overall_file, "w") as f:
        json.dump(all_folds_results, f, indent=2)

    print(f"\nNested CV completed. Results saved to {fold_out_dir}")


# -----------------------------------------------------------------------------
# Command line interface
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested cross‑validation for PointNet moisture classification")
    parser.add_argument("--veg", type=str, required=True, choices=["mushroom", "broccoli", "combine"],
                        help="Vegetable dataset")
    parser.add_argument("--channel", type=str, required=True, choices=["a", "b", "c"],
                        help="Channel name (CSV file name inside each sample folder)")
    parser.add_argument("--aug", action="store_true", help="Apply data augmentation")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--model", type=str, default="geo_int", choices=["geo", "geo_int"],
                        help="Model type: 'geo' (only geometry) or 'geo_int' (geometry + intensity)")
    parser.add_argument("--n_splits", type=int, default=10, help="Number of outer CV folds")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs per config")
    parser.add_argument("--data_root", type=str, default="./data", help="Root data directory")
    parser.add_argument("--results_root", type=str, default="./results", help="Root results directory")
    parser.add_argument("--fixed_lr", type=float, help="If provided, use fixed learning rate (skip inner search)")
    parser.add_argument("--fixed_dropout", type=float, help="If provided, use fixed dropout")
    parser.add_argument("--fixed_wd", type=float, help="If provided, use fixed weight decay")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Set device
    device = get_device(cuda_id=args.cuda) if "get_device" in globals() else torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    # Fixed parameters (if any)
    fixed_params = None
    if args.fixed_lr is not None or args.fixed_dropout is not None or args.fixed_wd is not None:
        fixed_params = {
            "lr": args.fixed_lr,
            "dropout": args.fixed_dropout,
            "weight_decay": args.fixed_wd,
        }
        # Ensure all are provided if any are given
        if None in fixed_params.values():
            raise ValueError("If using fixed parameters, must provide --fixed_lr, --fixed_dropout, and --fixed_wd")

    # Run nested CV
    nested_cross_validation(
        veg=args.veg,
        channel=args.channel,
        n_splits=args.n_splits,
        augmentation=args.aug,
        model_type=args.model,
        fixed_params=fixed_params,
        device=device,
        data_root=args.data_root,
        results_root=args.results_root,
        num_epochs=args.epochs,
    )


if __name__ == "__main__":
    main()