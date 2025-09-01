# VegPoint 🌱💧  
**From 3D Points to Drying Insight: Hierarchical Deep Learning on LiDAR-Based Geometry and Spectra**

[![Paper](https://img.shields.io/badge/Paper-arXiv-green)](https://)  
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  

This repository contains the code and dataset processing pipeline introduced in our paper:  

> **From 3D Points to Drying Insight: Hierarchical Deep Learning on LiDAR-Based Geometry and Spectra**  
> Majharulislam Babor, Arman Arefi, Barbara Sturm, Marina M.-C. Höhne, Manuela Zude-Sasse  
> Submitted to *___* (2025).  

---

## 🌟 Overview

Moisture content is a critical determinant of postharvest quality. Traditional sensing methods (e.g., hyperspectral, NIR) provide only 2D spectral information and fail to capture the **geometric heterogeneity** of agricultural products.  

**VegPoint** is the first annotated 3D LiDAR dataset for broccoli and mushroom, with per-segment (core vs. periphery) ground-truth moisture labels.  
We provide:  

- 🥦 **Dataset**: 3D LiDAR point clouds with intensity at 1320 nm, 1450 nm, and a dual-wavelength prototype (PT).  
- 🔧 **Pipeline**: Preprocessing (intensity smoothing, upright alignment, segmentation, augmentation).  
- 🤖 **Models**: PointNet++-based classifiers for geometry, intensity, and fused modalities.  
- 📊 **Evaluation**: 10-fold cross-validation, spatial transfer (core→periphery and vice versa), modality occlusion analysis.  

---

## 📂 Dataset

Each sample is represented as a point cloud with 5,000 points:  

- **Geometry**: 3D coordinates (x, y, z)  
- **Intensity**: Reflectance at each LiDAR wavelength  
- **Segments**: Core and periphery regions (aligned to ground-truth MC measurements)  
- **Moisture Classes**:  
  - High MC (> 88%)  
  - Medium MC (35–88%)  
  - Low MC (< 35%)  

👉 The dataset will be hosted [here](https://github.com/majharB/vegPoint/releases) (coming soon).  

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/majharB/vegPoint.git
cd vegPoint

# Create a conda environment
conda create -n vegpoint python=3.11
conda activate vegpoint

# Install requirements
pip install -r requirements.txt
