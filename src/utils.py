
import torch
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from pcdata import PcData, PcDataLoad
from collections import Counter
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from itertools import product
from augment import augment_point_cloud
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import json
import shutil
from torch.utils.data import DataLoader, WeightedRandomSampler
        
import warnings
warnings.filterwarnings("ignore")

    
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def safe_collate(batch):
    if len(batch) == 1:
        # Duplicate the only sample
        return torch.utils.data.dataloader.default_collate(batch * 2)
    else:
        return torch.utils.data.dataloader.default_collate(batch)
    
def split_data_cross_domain(split_method, channel, semi_supervised=False):
    """
    Splits the data into training, validation, and test sets for cross-domain evaluation.
    Allows optional semi-supervised setting where one sample per class is taken from the test domain.
    """
    folder_name_mushroom = './data/cloud/mushroom'
    folder_name_broccoli = './data/cloud/broccoli'
    ground_truth_file = './data/mc.csv'

    pcd_mushroom = PcData(folder_name_mushroom, ground_truth_file, channel_name=channel)
    pcd_broccoli = PcData(folder_name_broccoli, ground_truth_file, channel_name=channel)

    if split_method == 'mushroom_broccoli':
        pcd_train = pcd_mushroom
        pcd_test = pcd_broccoli
    elif split_method == 'broccoli_mushroom':
        pcd_train = pcd_broccoli
        pcd_test = pcd_mushroom
    else:
        raise ValueError(f"Invalid split method: {split_method}")

    # Start with all training data
    X_train_val, y_train_val = get_core_periphery(pcd_train, sample_idx=range(len(pcd_train.segmented_core)))

    # Handle semi-supervised by including 1 sample per class from test domain into training
    if semi_supervised:
        labels_test = np.array([i[1] for i in pcd_test.segmented_core]).astype(int)
        unique_classes = np.unique(labels_test)

        selected_indices = []
        for cls in unique_classes:
            class_indices = np.where(labels_test == cls)[0]
            if len(class_indices) == 0:
                raise ValueError(f"No samples found for class {cls} in test domain.")
            selected_indices.append(class_indices[0])  # Take first sample for class

        # Get 3 samples (1 per class) from test domain
        X_semi, y_semi = get_core_periphery(pcd_test, sample_idx=selected_indices)

        # Repeat them to match the size of training domain
        repeat_factor = int(np.ceil(len(X_train_val) / len(X_semi)))
        X_semi_repeated = np.tile(X_semi, (repeat_factor, 1, 1))[:len(X_train_val)]
        y_semi_repeated = np.tile(y_semi, (repeat_factor,))[:len(X_train_val)]

        # Concatenate to training set
        X_train_val = np.concatenate([X_train_val, X_semi_repeated], axis=0)
        y_train_val = np.concatenate([y_train_val, y_semi_repeated], axis=0)

        # Remove selected samples from test domain
        full_indices = np.arange(len(pcd_test.segmented_core))
        remaining_indices = np.setdiff1d(full_indices, selected_indices)
        X_test, y_test = get_core_periphery(pcd_test, sample_idx=remaining_indices)
    else:
        # Normal test set (no leakage)
        X_test, y_test = get_core_periphery(pcd_test, sample_idx=range(len(pcd_test.segmented_core)))

    # Stratified split for train/val
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(X_train_val)), y_train_val))
    X_train, y_train = X_train_val[train_idx], y_train_val[train_idx]
    X_val, y_val     = X_train_val[val_idx],   y_train_val[val_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test


def split_data(PC, method='random'):
    """
    Splits the dataset into training and validation sets while preventing data leakage.
    
    Parameters:
    - PC: PointCloudData object containing segmented core and periphery data.
    - val_ratio: Proportion of the dataset to include in the validation split.
    - seed: Random seed for reproducibility.
    
    Returns:
    - train_index: Indices for the training set.
    - test_index: Indices for the validation set.
    """
    # val_ratio = 0.03  # Proportion of the dataset to include in the validation split
    seed = 42  # Random seed for reproducibility
    if method == 'random':

        data = np.array([i[0] for i in PC.segmented_core])  # Extract point clouds
        labels = np.array([i[1] for i in PC.segmented_core])  # Extract
        
        # --- Step 1: Train+Val (80%) and Test (20%) Split ---
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_val_idx, test_idx = next(sss1.split(np.zeros(len(labels)), labels))

        # Subset the data
        train_val_data = data[train_val_idx]

        train_val_labels = labels[train_val_idx]

        # --- Step 2: Train (80%) and Val (20%) Split of Train+Val Set ---
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.06, random_state=42)  # 20% of 80% = 16% of total
        train_idx_rel, val_idx_rel = next(sss2.split(np.zeros(len(train_val_data)), train_val_labels))

        # Convert relative indices to absolute ones
        train_idx = train_val_idx[train_idx_rel]
        val_idx = train_val_idx[val_idx_rel]

        # Final splits
        X_train, y_train = get_core_periphery(PC, train_idx)
        X_val, y_val = get_core_periphery(PC, val_idx)
        X_test, y_test = get_core_periphery(PC, test_idx)
        return X_train, y_train, X_val, y_val, X_test, y_test
        
    elif method == 'core_periphery' or method == 'periphery_core':
        # Split data based on core and periphery
        X_core, y_core = [], []
        X_periphery, y_periphery = [], []
        for k in range(len(PC.segmented_core)):
            X, y = PC.segmented_core[k]
            X_core.append(X)
            y_core.append(y)
        for k in range(len(PC.segmented_periphery)):
            X, y = PC.segmented_periphery[k]
            X_periphery.append(X)
            y_periphery.append(y)
        X_core = np.array(X_core)
        y_core = np.array(y_core)
        X_periphery = np.array(X_periphery)
        y_periphery = np.array(y_periphery)
        
        if method == 'core_periphery':
            X_train_val = X_core.copy()
            y_train_val = y_core.copy()
            X_test = X_periphery.copy()
            y_test = y_periphery.copy()
        else:  # method == 'periphery_core'
            X_train_val = X_periphery.copy()
            y_train_val = y_periphery.copy()
            X_test = X_core.copy()
            y_test = y_core.copy()
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=42)  
        train_idx, val_idx = next(sss.split(np.zeros(len(X_train_val)), y_train_val))
        X_train, y_train = X_train_val[train_idx], y_train_val[train_idx]
        X_val, y_val     = X_train_val[val_idx],   y_train_val[val_idx]
        return X_train, y_train, X_val, y_val, X_test, y_test
        
    else:
        raise ValueError("Invalid method. Choose 'random', 'core_periphery', or 'periphery_core'.")




def get_core_periphery(PC, sample_idx):
    """
    Get core and periphery data for a specific sample index.
    
    Parameters:
    - PC: PointCloudData object containing segmented core and periphery data.
    - sample_idx: Index of the sample to retrieve.
    
    Returns:
    - core_points: Core points of the sample.
    - periphery_points: Periphery points of the sample.
    """
    X, y = [], []
    for k in sample_idx:
        core_points, core_label = PC.segmented_core[k]
        X.append(core_points)
        y.append(core_label)
        
        periphery_points, periphery_label = PC.segmented_periphery[k]
        X.append(periphery_points)
        y.append(periphery_label)
    return np.array(X), np.array(y)


def prevent_data_leak(PC, train_index, test_index):
    X_train, y_train = [], []
    X_test, y_test = [],[]
    
    for k in train_index:
        X, y = PC.segmented_core[k]
        X_train.append(X)
        y_train.append(y)
        
        X, y = PC.segmented_periphery[k]
        X_train.append(X)
        y_train.append(y)
        
    for k in test_index:
        X, y = PC.segmented_core[k]
        X_test.append(X)
        y_test.append(y)
        
        X, y = PC.segmented_periphery[k]
        X_test.append(X)
        y_test.append(y)
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
    
        
def get_device(cuda_id=0):
    if torch.backends.mps.is_available():
        print("Using device: MPS (Metal Performance Shaders)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using device: CUDA")
        return torch.device(f"cuda:{cuda_id}")
    else:
        print("Using device: CPU")
        return torch.device("cpu")

def create_oversampler(y):
    """
    Create a WeightedRandomSampler to oversample underrepresented classes.
    
    Args:
        y (array-like): Class labels (numpy array or list).
    
    Returns:
        sampler (WeightedRandomSampler): Sampler to use in DataLoader.
    """
    y = np.array(y).astype(int)
    class_counts = np.bincount(y)
    class_weights = 1. / class_counts
    sample_weights = class_weights[y]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(y),
        replacement=True
    )
    return sampler

import numpy as np

def create_balanced_dataset(X, y):
    X = np.array(X)
    y = np.array(y)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_balanced = []
    y_balanced = []

    for cls in classes:
        idx = np.where(y == cls)[0]
        num_repeat = max_count // len(idx)
        remainder = max_count % len(idx)

        # Oversample with replacement
        idx_repeated = np.tile(idx, num_repeat)
        idx_extra = np.random.choice(idx, remainder, replace=True)
        idx_total = np.concatenate([idx_repeated, idx_extra])

        X_balanced.append(X[idx_total])
        y_balanced.append(y[idx_total])

    X_balanced = np.concatenate(X_balanced, axis=0)
    y_balanced = np.concatenate(y_balanced, axis=0)
    
    # Shuffle
    indices = np.random.permutation(len(y_balanced))
    return X_balanced[indices], y_balanced[indices]


def check_class_distribution(loader):
    """
    Prints the number of samples per class in a PyTorch DataLoader.
    Works with loaders using oversampling or regular sampling.
    """
    sampled_labels = []

    for _, labels in loader:
        # Flatten the labels and convert to list
        if isinstance(labels, torch.Tensor):
            labels = labels.view(-1).tolist()
        else:
            labels = list(labels)
        sampled_labels.extend(labels)

    class_counts = Counter(sampled_labels)

    print("📊 Class distribution:")
    for cls in sorted(class_counts):
        print(f"  Class {cls}: {class_counts[cls]} samples")