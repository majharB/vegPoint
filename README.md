# VegPoint 🌱💧
**Postharvest Moisture Prediction using LiDAR Point Clouds: Integrating 3D Structure and Spectral Intensity**

[![Paper](https://img.shields.io/badge/Paper-arXiv-green)](https://arxiv.org/abs/XXXX.XXXX) (coming soon)
[![Dataset](https://img.shields.io/badge/DOI-10.5281/zenodo.19001042-blue)](https://doi.org/10.5281/zenodo.19001042)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the code and dataset processing pipeline for our paper:

> **From 3D Points to Drying Insight: Hierarchical Deep Learning on LiDAR-Based Geometry and Spectra**  
> Majharulislam Babor, Arman Arefi, Barbara Sturm, Marina M.-C. Höhne, Manuela Zude-Sasse  
> *Submitted, 2025*

---

## 🌟 Overview

Moisture content is a critical quality parameter in postharvest storage. Traditional spectral sensors capture only 2D information and miss the **3D geometric heterogeneity** of produce.  
**VegPoint** is the first annotated 3D LiDAR dataset for broccoli and mushroom, providing:

- **3D point clouds** with intensity at 1320 nm, 1450 nm, and a dual‑wavelength moisture index (MI).
- **Per‑segment** moisture labels for core and periphery regions (high/medium/low classes).
- A complete pipeline for preprocessing, augmentation, and deep learning with PointNet++.

Our work demonstrates that combining geometry and intensity improves moisture classification, especially under spatial transfer (core→periphery) and when one modality is occluded.

---

## 📂 Dataset

Each sample is a point cloud of **5,000 points** (after resizing) with the following attributes:

| Field       | Description                                      |
|-------------|--------------------------------------------------|
| `x, y, z`   | 3D coordinates (normalized per segment)          |
| `intensity` | Reflectance at the given LiDAR wavelength (1320 nm, 1450 nm or moisture index MI)       |
| `region`    | Core or periphery (derived from XY distance)     |
| `moisture`  | Class: 0 (low, <35%), 1 (medium, 35‑88%), 2 (high, >88%) |

The dataset is available at Zenodo:

🔗 [**10.5281/zenodo.19001042**](https://doi.org/10.5281/zenodo.19001042)

After downloading, place the contents in `data/cloud/` following this structure:

```bash
data/
├── cloud/
│   ├── mushroom/
│   │   ├── sample1/
│   │   │   ├── a.csv # x,y,z and intensity at wavelength 1320 nm
│   │   │   ├── b.csv # x,y,z and intensity at wavelength 1450 nm
│   │   │   └── c.csv # x,y,z and intensity at moisture index (MI)
│   │   ├── sample2/
│   │   └── ...
│   └── broccoli/
│       ├── sample1/
│       └── ...
└── mc.csv # ground-truth moisture per sample & region
```

The file `mc.csv` must contain at least the columns: `sample_id`, `moisture_core`, `moisture_periphery`.

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/majharB/vegPoint.git
cd vegPoint
```
### 2. Set up a virtual environment (conda or venv)
```
conda create -n vegpoint python=3.11
conda activate vegpoint
# or
python -m venv vegpoint
```
### 3. Install dependencies
```
pip install -r requirements.txt
```
If you use a GPU, ensure PyTorch with CUDA is installed separately (see pytorch.org).

🚀 Usage
All experiments are organized as scripts in the scripts/ directory. The core modules are in src/ (package name pcdg).

Data preparation
The dataset is automatically loaded and preprocessed by src/pcdata.py when you run any training script. No manual preprocessing is needed.

### 4. Training a model
#### 4a. standard random split
Train a PointNet++ model on a broccoli/ mushrooms and wavelength intensity with a simple train/val/test split:

```
source vegpoint/bin/activate
python scripts/train.py \
    --veg mushroom \
    --channel a \
    --split_method random \
    --aug \
    --epochs 30 \
    --cuda 0
```
Options:

--veg: mushroom, broccoli, or combine

--channel: a, b, or c (intensity option)

--split_method: random, core_periphery (train on core, test on periphery), periphery_core (reverse)

--aug: enable data augmentation (rotation, scaling, jittering)

--semi_supervised: (for spatial transfer) include a few labeled samples from the test domain in training (no data from target region was used in paper)

--model: geo (only geometry) or geo_int (geometry + intensity)

--epochs, --cuda, etc.

#### 4b. Nested cross‑validation

To perform 10‑fold nested CV with hyperparameter tuning (as used in the paper):

```
python scripts/cross_val.py \
    --veg mushroom \
    --channel a \
    --aug \
    --model geo_int \
    --cuda 0 \
    --n_splits 10 \
    --epochs 30
```

The script saves per‑fold results (test indices, best config, losses, predictions) in results/folds/<veg>/. The inner loop trains each configuration for 10 epochs; the best configuration is then evaluated on the test set.

#### 4c. Spatial transfer experiments
Train on one region and evaluate on the other (core→periphery or periphery→core) as reported in the paper:
```
python scripts/train_spatial_transfer.py \
    --veg broccoli \
    --channel b \
    --split_method core_periphery \
    --aug \
    --semi_supervised \
    --model geo_int \
    --epochs 30 \
    --cuda 0
```

## 📝 Citation

BibTeX
```
@article{babor_2026_vegpoint,
  title={From 3D Points to Drying Insight: Hierarchical Deep Learning on LiDAR-Based Geometry and Spectra},
  author={Babor, M. and Arefi, A. and Sturm, B. and Höhne, M. M.-C. and Zude-Sasse, M.},
  journal={},
  year={2026}
}
```
APA
```
Babor, M., Arefi, A., Sturm, B., Höhne, M. M.-C., & Zude-Sasse, M. (2025). From 3D points to drying insight: Hierarchical deep learning on LiDAR-based geometry and spectra. Manuscript submitted for publication.
```
