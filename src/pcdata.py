"""
Point cloud data loading and preprocessing for wood moisture content prediction.
Handles raw CSV files, applies intensity smoothing, PCA alignment, segmentation
into core/periphery, and resizes all clouds to a uniform number of points.
"""

import os
import re
from typing import List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset


class PcData:
    """
    Loads point cloud data from a folder structure, applies preprocessing,
    and produces a unified dataset of core and periphery segments with labels.

    The folder should contain subfolders named like 'sample1', 'sample2', etc.
    Each subfolder must contain a CSV file named '{channel_name}.csv' with columns:
    x, y, z, intensity.

    A separate ground truth CSV file must contain at least:
    sample_id, moisture_core, moisture_periphery.

    Processing steps:
        1. Intensity smoothing (spatial Gaussian filter)
        2. Upright alignment using PCA (principal component aligned with z‑axis)
        3. Global normalization of coordinates (centering and scaling to [-1,1])
        4. Segmentation into core (central cylinder) and periphery based on `core_ratio`
        5. Per‑segment recentering and scaling to unit sphere
        6. Resizing all clouds to a common number of points (`target_size`)
        7. Combining core and periphery clouds into one dataset with labels
    """

    def __init__(
        self,
        folder_name: str,
        ground_truth_file: str,
        channel_name: str = 'a',
        core_ratio: float = 0.5,
        filter_sigma: float = 10.0,
        recenter: bool = True,
        target_size: Optional[int] = None,
        moisture_bins: Tuple[float, float] = (0.35, 0.88)
    ):
        """
        Args:
            folder_name: Path to the folder containing sample subfolders.
            ground_truth_file: Path to CSV with columns sample_id, moisture_core, moisture_periphery.
            channel_name: Base name of the CSV file inside each subfolder (e.g., 'a' -> 'a.csv').
            core_ratio: Fraction of the XY extent used to define the core radius.
            filter_sigma: Spatial sigma for Gaussian intensity smoothing (in same units as XYZ).
            recenter: If True, apply per‑segment recentering + scaling to unit sphere.
            target_size: Desired number of points per cloud. If None, uses the minimum size
                         across all clouds (default behaviour).
            moisture_bins: (low_threshold, high_threshold) to discretize moisture into
                           three classes: 0 (low), 1 (medium), 2 (high).
        """
        self.folder_name = folder_name
        self.ground_truth_df = pd.read_csv(ground_truth_file)
        self.channel_name = channel_name
        self.core_ratio = core_ratio
        self.filter_sigma = filter_sigma
        self.recenter = recenter
        self.moisture_bins = moisture_bins
        self.target_size = target_size

        # Internal storage
        self.segmented_core: List[Tuple[np.ndarray, int]] = []   # (points, label)
        self.segmented_periphery: List[Tuple[np.ndarray, int]] = []
        self.data: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None

        # Run the full pipeline
        self._process_all_samples()
        self._resize_to_common_size()
        self._combine()

    def _process_all_samples(self) -> None:
        """Iterate over all sample subfolders and process each one."""
        subfolders = sorted(
            [f for f in os.listdir(self.folder_name)
             if os.path.isdir(os.path.join(self.folder_name, f))],
            key=self._split_letters_numbers
        )
        for subfolder in subfolders:
            self._process_single_sample(subfolder)

    def _process_single_sample(self, sample_id: str) -> None:
        """Load, preprocess, and segment a single sample."""
        csv_path = os.path.join(self.folder_name, sample_id, f"{self.channel_name}.csv")
        if not os.path.exists(csv_path):
            return

        # Load raw points
        points = pd.read_csv(csv_path)[['x', 'y', 'z', 'intensity']].values

        # Preprocessing pipeline
        points = self._filter_intensity(points, sigma=self.filter_sigma)
        points = self._upright_alignment(points)
        points = self._normalize_coordinates(points)      # global normalization
        core, periphery = self._segment_core_periphery(points)

        # Per‑segment recentering (optional)
        if self.recenter:
            core = self._recenter_segment(core)
            periphery = self._recenter_segment(periphery)

        # Get moisture labels (discretized)
        label_core = self._get_moisture_class(sample_id, 'core')
        label_periphery = self._get_moisture_class(sample_id, 'periphery')

        self.segmented_core.append((core, label_core))
        self.segmented_periphery.append((periphery, label_periphery))

    # -------------------------------------------------------------------------
    # Preprocessing steps
    # -------------------------------------------------------------------------
    @staticmethod
    def _filter_intensity(points: np.ndarray, sigma: float, k: int = 50) -> np.ndarray:
        """
        Smooth intensity values using a spatial Gaussian kernel (k‑NN).
        Modifies the intensity column in place but returns the whole array.
        """
        xyz = points[:, :3].astype(np.float64)
        I = points[:, 3].astype(np.float64)

        k = min(k, len(points))
        nn = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(xyz)
        idx = nn.kneighbors(return_distance=False)          # [N, k]

        # squared distances
        d2 = ((xyz[:, None, :] - xyz[idx]) ** 2).sum(axis=2)
        w = np.exp(-d2 / (2.0 * sigma ** 2 + 1e-12))
        w /= (w.sum(axis=1, keepdims=True) + 1e-12)

        I_smooth = (w * I[idx]).sum(axis=1)
        points[:, 3] = I_smooth
        return points

    @staticmethod
    def _upright_alignment(points: np.ndarray) -> np.ndarray:
        """
        Rotate the cloud so that the principal component (PC1) aligns with the z‑axis.
        Returns cloud with same columns (x,y,z,intensity).
        """
        has_intensity = points.shape[1] == 4
        xyz = points[:, :3]
        centroid = xyz.mean(axis=0)
        pca = PCA(n_components=3)
        pca.fit(xyz)
        # The rotation matrix that aligns PC1 with z: we simply use pca.components_.T?
        # Actually pca.components_ are the directions (row = component). To align data,
        # we project onto these components: (xyz - centroid) @ pca.components_.T
        # This gives coordinates in the PCA basis. We want PC1 to be the new z,
        # so we reorder axes? But the code in original simply did:
        # aligned_xyz = (xyz - centroid) @ rotation_matrix.T  where rotation_matrix = pca.components_
        # That projects onto the principal components, but the resulting axes are in order of variance.
        # It does not explicitly align PC1 with z; the cloud is just rotated to its eigenbasis.
        # We keep the original behaviour.
        aligned_xyz = (xyz - centroid) @ pca.components_.T
        if has_intensity:
            return np.column_stack((aligned_xyz, points[:, 3]))
        else:
            return aligned_xyz

    @staticmethod
    def _normalize_coordinates(points: np.ndarray) -> np.ndarray:
        """Globally center and scale coordinates to [-1, 1] range per axis."""
        centered = points[:, :3] - points[:, :3].mean(axis=0)
        scale = np.max(np.abs(centered), axis=0)
        # Avoid division by zero
        scale[scale == 0] = 1.0
        points[:, :3] = centered / scale
        return points

    def _segment_core_periphery(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split cloud into core (central cylinder) and periphery based on XY distance.
        Core radius = core_ratio * (half of the smaller XY extent).
        """
        xmin, xmax = points[:, 0].min(), points[:, 0].max()
        ymin, ymax = points[:, 1].min(), points[:, 1].max()
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        radius = self.core_ratio * min(xmax - xmin, ymax - ymin) / 2.0

        distances = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
        core_mask = distances <= radius
        return points[core_mask], points[~core_mask]

    @staticmethod
    def _recenter_segment(points: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Per‑segment normalization: center and scale so that points lie within a unit sphere.
        """
        centered = points[:, :3] - points[:, :3].mean(axis=0)
        max_norm = np.linalg.norm(centered, axis=1).max()
        if max_norm > eps:
            centered /= max_norm
        points[:, :3] = centered
        return points

    # -------------------------------------------------------------------------
    # Label handling
    # -------------------------------------------------------------------------
    def _get_moisture_class(self, sample_id: str, region: str) -> int:
        """
        Retrieve moisture value for the given sample and region, and discretize
        into three classes based on self.moisture_bins.
        """
        col = f'moisture_{region}'
        mc = self.ground_truth_df.loc[self.ground_truth_df['sample_id'] == sample_id, col].values[0]
        low, high = self.moisture_bins
        if mc < low:
            return 0
        elif mc < high:
            return 1
        else:
            return 2

    # -------------------------------------------------------------------------
    # Resizing and combination
    # -------------------------------------------------------------------------
    def _resize_to_common_size(self) -> None:
        """Resize all core and periphery clouds to the same number of points."""
        if self.target_size is None:
            # Determine minimum size across all clouds
            core_sizes = [seg.shape[0] for seg, _ in self.segmented_core]
            peri_sizes = [seg.shape[0] for seg, _ in self.segmented_periphery]
            self.target_size = min(min(core_sizes), min(peri_sizes))
            # Optional: enforce a lower bound (e.g., 5000) as in original code
            # self.target_size = max(self.target_size, 5000)

        # Resize core
        resized_core = []
        for seg, label in self.segmented_core:
            resized = self._resize_point_cloud(seg, self.target_size)
            resized_core.append((resized, label))
        self.segmented_core = resized_core

        # Resize periphery
        resized_peri = []
        for seg, label in self.segmented_periphery:
            resized = self._resize_point_cloud(seg, self.target_size)
            resized_peri.append((resized, label))
        self.segmented_periphery = resized_peri

    @staticmethod
    def _resize_point_cloud(points: np.ndarray, target_size: int) -> np.ndarray:
        """
        Randomly sample (with replacement if needed) to achieve exactly target_size points.
        """
        current = points.shape[0]
        if current == target_size:
            return points.copy()
        elif current < target_size:
            # oversample with replacement
            indices = np.random.choice(current, target_size, replace=True)
        else:
            # subsample without replacement
            indices = np.random.choice(current, target_size, replace=False)
        return points[indices].copy()

    def _combine(self) -> None:
        """Concatenate all core and periphery clouds and their labels."""
        all_data = []
        all_labels = []
        for seg, label in self.segmented_core:
            all_data.append(seg)
            all_labels.append(label)
        for seg, label in self.segmented_periphery:
            all_data.append(seg)
            all_labels.append(label)

        self.data = np.array(all_data, dtype=object)  # keep as list of arrays
        self.labels = np.array(all_labels, dtype=np.int64)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    @staticmethod
    def _split_letters_numbers(name: str) -> Tuple[Union[str, int], ...]:
        """
        Sort subfolders naturally (e.g., sample1, sample2, ...).
        Used as key for sorted().
        """
        match = re.match(r"([a-zA-Z]+)(\d+)", name)
        if match:
            letters, numbers = match.groups()
            return (letters, int(numbers))
        return (name, 0)


class PcDataset(Dataset):
    """PyTorch Dataset for point clouds and corresponding moisture classes."""

    def __init__(self, data: np.ndarray, labels: np.ndarray):
        """
        Args:
            data: List/array of point clouds (each shape (N, 4)).
            labels: 1D array of integer labels (0,1,2).
        """
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cloud = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return cloud, label


# Example usage (commented out – move to a separate script)
# if __name__ == "__main__":
#     loader = PcData(
#         folder_name='../data/cloud',
#         ground_truth_file='./data/mc.csv',
#         channel_name='a',
#         core_ratio=0.5,
#         filter_sigma=10.0,
#         recenter=True
#     )
#     dataset = PcDataset(loader.data, loader.labels)
#     print(f"Dataset size: {len(dataset)}")