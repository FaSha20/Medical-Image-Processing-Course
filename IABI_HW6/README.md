# Vision Transformer (ViT) for Brain Tumor Image Classification

This repository contains the Jupyter notebook [ViT_Classification+Visualization.ipynb](ViT_Classification+Visualization.ipynb), which demonstrates the use of Vision Transformers (ViT) for classifying brain MRI images into four categories. The notebook also includes visualization of the model's attention maps to provide interpretability for its predictions.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Notebook Structure](#notebook-structure)
- [How to Run](#how-to-run)
- [Key Components](#key-components)
- [Results & Visualization](#results--visualization)
- [Requirements](#requirements)
- [References](#references)

---

## Overview

Vision Transformers (ViT) adapt the Transformer architecture, originally developed for natural language processing, to image classification tasks. Instead of using convolutional layers, ViT splits images into patches, embeds them, and processes them as sequences using self-attention. This notebook applies ViT to the Brain Tumor MRI dataset, classifying images into four classes: glioma, meningioma, notumor, and pituitary.

The notebook covers:
- Data preparation and augmentation
- Custom implementation of ViT from scratch in PyTorch
- Model training and evaluation
- Visualization of attention maps for interpretability

---

## Dataset

The Brain Tumor dataset consists of MRI scans categorized into four classes. The notebook downloads and extracts the dataset automatically:

```python
!gdown 1nSwAxOS2bCn9m7cXnA-z9-0ADhb4CBuv
!unzip Brain_tumor.zip
```

- Training images: `/content/Training`
- Testing images: `/content/Testing`

Each subfolder corresponds to a class label.

---

## Notebook Structure

1. **Introduction**: Background on ViT and the Brain Tumor dataset.
2. **Imports & Setup**: Loads required libraries and sets hyperparameters.
3. **Dataset Preparation**: Applies data augmentation and normalization, loads datasets, and creates data loaders.
4. **Model Definition**: Implements Patch Embedding, Transformer Encoder, and the full Vision Transformer architecture from scratch.
5. **Training Loop**: Trains the ViT model, tracks loss and accuracy, and saves the best model.
6. **Evaluation**: Evaluates the model on the test set and computes accuracy.
7. **Visualization**: Plots training/validation loss and accuracy, confusion matrix, and attention maps for interpretability.

---

## How to Run

1. **Clone the repository** and open [ViT_Classification+Visualization.ipynb](ViT_Classification+Visualization.ipynb) in Jupyter Notebook or VS Code.
2. **Install dependencies** (see [Requirements](#requirements)).
3. **Run all cells** in order. The notebook will:
   - Download and extract the dataset.
   - Prepare data loaders.
   - Train the Vision Transformer model.
   - Evaluate and visualize results.

---

## Key Components

- **PatchEmbedding**: Splits images into patches and embeds them for the transformer.
- **TransformerEncoderLayer & TransformerEncoder**: Implements multi-head self-attention and MLP layers.
- **VisionTransformer**: Full ViT model with class token and positional embeddings.
- **Training & Evaluation**: Standard PyTorch training loop with validation and test evaluation.
- **Visualization**:
  - Training/validation loss and accuracy plots
  - Confusion matrix for test set
  - Attention map visualization for interpretability

---

## Results & Visualization

- **Training and validation metrics** are plotted over epochs.
- **Best model checkpoint** is saved as `best_vit_model.pth`.
- **Test set evaluation** reports final accuracy.
- **Confusion matrix** visualizes class-wise performance.
- **Attention maps** show which image regions the ViT model focuses on for its predictions.

---

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- numpy
- matplotlib
- seaborn
- scikit-learn
- gdown

Install dependencies with:

```sh
pip install torch torchvision numpy matplotlib seaborn scikit-learn gdown
```

---

## References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT Paper)](https://arxiv.org/abs/2010.11929)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Original Brain Tumor Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

