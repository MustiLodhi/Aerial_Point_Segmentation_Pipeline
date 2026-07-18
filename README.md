# 🌍 Aerial Point Segmentation Pipeline

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Google Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/)

A deep learning framework designed for semantic segmentation on remote sensing and aerial imagery. Instead of relying on expensive, densely annotated masks, this pipeline trains a U-Net architecture using incomplete (sparse) point-based annotations, effectively extracting high-resolution spatial features from minimal data.

## 🚀 Project Overview

Accurate extraction of building footprints and infrastructure from aerial imagery is a core challenge in geoinformatics and spatial analysis. This project tackles the bottleneck of dataset annotation by implementing a strategy that learns from sparse points rather than full polygon masks.

**Key Methodology:**
* **Environment:** Developed and executed entirely in Google Colab.
* **Architecture:** U-Net with a pre-trained ResNet-34 encoder (Transfer Learning via `segmentation-models-pytorch`).
* **Data Strategy:** Utilizes the MapAI Aerial Dataset. A custom `RealAerialPointDataset` simulates incomplete tagging by sampling *N* points per class, assigning the remainder an `ignore_index=255`.
* **Loss Function:** Partial Multiclass Focal Loss, which masks background pixels to prevent unannotated regions from skewing the gradient calculations.
* **Experiments:** Evaluated model convergence and boundary precision under "Low Sparsity" (20 points/class) and "High Sparsity" (100 points/class) scenarios.

## 📊 Inference Visualizations

The panels below illustrate the model performance on test data.

**Columns (Left to Right):** RGB Input | Sparse Annotations | Dense Ground Truth | Model Prediction

### Low Sparsity: 20 Points Per Class
<p align="center">
  <img src="1.png" width="80%">
  <br>
  <img src="2.png" width="80%">
  <br>
  <img src="3.png" width="80%">
</p>

### High Sparsity: 100 Points Per Class
<p align="center">
  <img src="4.png" width="80%">
  <br>
  <img src="5.png" width="80%">
  <br>
  <img src="6.png" width="80%">
</p> 

## 🔑 Setup & Authentication

This pipeline programmatically downloads the dataset from Hugging Face. To run the notebook in Google Colab, you must provide your own Hugging Face Access Token.

**In Google Colab:**
1. Open the **Secrets** tab (the 🔑 icon on the left sidebar).
2. Create a new secret named `HF_TOKEN`.
3. Paste your Hugging Face Access Token into the value field.
4. Enable the **"Notebook access"** toggle for the secret.

## 💻 Installation

The notebook requires the following dependencies. Run this block in your first Colab cell to set up the environment:

```bash
!pip install torch torchvision datasets segmentation-models-pytorch
