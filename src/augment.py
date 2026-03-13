"""
Point cloud augmentation utilities for research.
Provides geometric transformations (rotation, scaling, jittering) and
dataset balancing via oversampling of underrepresented target bins.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import Counter
from sklearn.preprocessing import KBinsDiscretizer
from typing import List, Tuple, Optional

# For reproducibility, but can be set elsewhere if needed
# import os
# os.environ["OMP_NUM_THREADS"] = "1"


class PointAugment:
    """
    Applies various augmentations to a single point cloud.
    Assumes input shape (N, 4) with columns: x, y, z, intensity.
    All methods return a new array without modifying the original.
    """

    def __init__(self, rotation_range: float = 180, noise_std: float = 0.001, dropout_rate: float = 0.1):
        """
        Args:
            rotation_range: Maximum rotation angle in degrees (around z-axis).
            noise_std: Standard deviation of Gaussian noise added to coordinates.
            dropout_rate: Fraction of points to randomly drop (not implemented yet).
        """
        self.rotation_range = rotation_range
        self.noise_std = noise_std
        self.dropout_rate = dropout_rate

    def random_rotation(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        Randomly rotate the point cloud around its centroid (z‑axis only).

        Args:
            point_cloud: Array of shape (N, 4).

        Returns:
            Rotated point cloud (same shape).
        """
        # Work on a copy to avoid modifying original
        pc = point_cloud.copy()
        centroid = np.mean(pc[:, :3], axis=0)
        centered = pc[:, :3] - centroid

        angle = np.random.uniform(-self.rotation_range, self.rotation_range) * np.pi / 180
        rotation = R.from_euler('z', angle)
        rotated = rotation.apply(centered) + centroid

        # Reattach intensity
        result = np.column_stack((rotated, pc[:, 3]))
        return result

    def object_scaling(self, point_cloud: np.ndarray,
                       min_scale: float = 0.9, max_scale: float = 1.1) -> np.ndarray:
        """
        Scale x,y,z coordinates by a random factor uniformly (around origin).

        Args:
            point_cloud: Array of shape (N, 4).
            min_scale: Minimum scaling factor.
            max_scale: Maximum scaling factor.

        Returns:
            Scaled point cloud.
        """
        pc = point_cloud.copy()
        scale = np.random.uniform(min_scale, max_scale)
        pc[:, :3] *= scale
        return pc

    def shuffle_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        Randomly shuffle the order of points.

        Args:
            point_cloud: Array of shape (N, 4).

        Returns:
            Shuffled point cloud.
        """
        indices = np.random.permutation(point_cloud.shape[0])
        return point_cloud[indices].copy()

    def jittering(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        Add independent Gaussian noise to each point's coordinates.

        Args:
            point_cloud: Array of shape (N, 4).

        Returns:
            Jittered point cloud.
        """
        pc = point_cloud.copy()
        noise = np.random.normal(0, self.noise_std, pc[:, :3].shape)
        pc[:, :3] += noise
        return pc

    def augment(self, point_cloud: np.ndarray,
                operations: Optional[List[str]] = None,
                repeats: int = 3) -> List[np.ndarray]:
        """
        Generate multiple augmented versions of a point cloud.

        Args:
            point_cloud: Input cloud of shape (N, 4).
            operations: List of augmentation names to apply.
                        If None, uses ['rotation', 'scaling', 'jittering'].
            repeats: Number of times to generate each operation (total = repeats * len(operations)).

        Returns:
            List of augmented point clouds.
        """
        if operations is None:
            operations = ['rotation', 'scaling', 'jittering']

        augmented = []
        for _ in range(repeats):
            for op in operations:
                if op == 'rotation':
                    aug = self.random_rotation(point_cloud)
                elif op == 'scaling':
                    aug = self.object_scaling(point_cloud)
                elif op == 'jittering':
                    aug = self.jittering(point_cloud)
                # Add more operations (e.g., 'shuffle', 'dropout') as needed
                else:
                    raise ValueError(f"Unknown operation: {op}")
                augmented.append(aug)
        return augmented


def get_target_weights(y: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """
    Compute sample weights for balancing a regression target via binning.

    Args:
        y: 1D array of target values.
        n_bins: Number of bins for uniform discretization.

    Returns:
        Array of weights (larger for underrepresented bins).
    """
    y = np.array(y).reshape(-1, 1)
    kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    bin_indices = kb.fit_transform(y).astype(int).flatten()
    bin_counts = Counter(bin_indices)
    max_count = max(bin_counts.values())
    weights = np.array([max_count / bin_counts[idx] for idx in bin_indices])
    return weights


def get_augmentation_profile(y: np.ndarray,
                             total_augmented: int = 1000,
                             n_bins: int = 20) -> np.ndarray:
    """
    Determine how many times each original sample should be augmented
    to achieve a balanced dataset of size `total_augmented`.

    Args:
        y: 1D array of target values.
        total_augmented: Desired total number of samples after augmentation.
        n_bins: Number of bins for target weighting.

    Returns:
        Integer array of length len(y) with augmentation counts per sample.
    """
    y = np.array(y)
    weights = get_target_weights(y, n_bins)
    num_original = len(y)
    num_augmented = int(total_augmented - num_original)

    # Sampling probabilities proportional to weights
    probs = weights / weights.sum()
    aug_indices = np.random.choice(num_original, size=num_augmented, p=probs)

    # Count selections per original index
    counts = np.zeros(num_original, dtype=int)
    for idx in aug_indices:
        counts[idx] += 1
    return counts


def augment_point_cloud(X, y, total_augmented: int = 1000,
                        n_bins: int = 20,
                        augmenter: Optional[PointAugment] = None) -> Tuple[list, np.ndarray]:
    """
    Augment a collection of point clouds to balance the target distribution.

    Args:
        X: List of point clouds (each a numpy array of shape (N_i, 4)).
        y: 1D array of target values for each cloud.
        total_augmented: Desired total number of clouds after augmentation.
        n_bins: Number of bins for target weighting.
        augmenter: PointAugment instance. If None, a default one is created.

    Returns:
        X_aug: List of augmented point clouds (including originals).
        y_aug: Corresponding target values (numpy array).
    """
    y = np.array(y)
    counts = get_augmentation_profile(y, total_augmented, n_bins)

    if augmenter is None:
        augmenter = PointAugment()

    X_aug = []
    y_aug = []

    for i, cloud in enumerate(X):
        # Keep original
        X_aug.append(cloud)
        y_aug.append(y[i])

        # Generate required number of augmented versions
        num_needed = counts[i]
        if num_needed > 0:
            # For simplicity, we generate all at once; could be more efficient
            # by reusing augmentations, but this is clear.
            for _ in range(num_needed):
                # Apply a random sequence of augmentations (here just one random op)
                # For more variety, you could combine multiple ops.
                op = np.random.choice(['rotation', 'scaling', 'jittering'])
                aug_cloud = getattr(augmenter, f'random_{op}' if op != 'scaling' else 'object_scaling')(cloud)
                X_aug.append(aug_cloud)
                y_aug.append(y[i])

    return X_aug, np.array(y_aug)