# U-Net Image Segmentation for Lung Masks

This repository contains a Jupyter notebook, [Segmentation_UNET.ipynb](Segmentation_UNET.ipynb), that implements the U-Net architecture for medical image segmentation, specifically for lung mask extraction from chest X-ray images. The notebook guides you through data preparation, model building, training, evaluation, and visualization.

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

U-Net is a convolutional neural network architecture designed for fast and precise image segmentation. This notebook demonstrates how to:

- Prepare and preprocess a medical image dataset.
- Implement custom PyTorch datasets and data augmentations.
- Build and train a U-Net model (with an option for a pretrained VGG-11 encoder).
- Evaluate segmentation performance using metrics like Dice and Jaccard (IoU).
- Visualize predictions and compare them to ground truth masks.

## Dataset

The dataset consists of chest X-ray images and corresponding binary lung masks. The notebook downloads and extracts the dataset automatically using the following commands:

```python
!gdown "https://drive.google.com/uc?id=1ffbbyoPf-I3Y0iGbBahXpWqYdGd7xxQQ" -O dataset.tar.gz
!tar -xf dataset.tar.gz
```

- Images are stored in `dataset/images/`
- Masks are stored in `dataset/masks/`

## Notebook Structure

1. **Introduction**: Explains the purpose and objectives of the notebook.
2. **Imports & Setup**: Loads required libraries and sets up the environment.
3. **Data Preparation**: Downloads, extracts, and splits the dataset into training, validation, and test sets.
4. **Custom Dataset & Augmentations**: Implements `LungDataset` and custom transforms (`Pad`, `Crop`, `Resize`).
5. **Visualization Utilities**: Provides functions to blend and visualize images and masks.
6. **Model Definition**: Implements the U-Net architecture and a variant with a pretrained VGG-11 encoder.
7. **Training Loop**: Trains the model, logs metrics, and saves the best checkpoint.
8. **Evaluation**: Evaluates the model on the test set and visualizes predictions.
9. **Helper Functions**: Includes metric calculations (Dice, Jaccard).

## How to Run

1. **Clone the repository** and open [Segmentation_UNET.ipynb](Segmentation_UNET.ipynb) in Jupyter Notebook or VS Code.
2. **Install dependencies** (see [Requirements](#requirements)).
3. **Run all cells** in order. The notebook will:
   - Download and extract the dataset.
   - Prepare data loaders.
   - Train the U-Net model.
   - Evaluate and visualize results.

## Key Components

- **LungDataset**: Custom PyTorch dataset for loading and preprocessing image-mask pairs.
- **Pad, Crop, Resize**: Custom data augmentation transforms for robust training.
- **UNet & PretrainedUNet**: U-Net architectures, with the latter using a VGG-11 encoder for improved performance.
- **blend**: Utility for overlaying masks on images for qualitative analysis.
- **Metrics**: Dice and Jaccard (IoU) for quantitative evaluation.

## Results & Visualization

- **Training and validation metrics** (loss, Dice, Jaccard) are plotted over epochs.
- **Best model checkpoint** is saved as `best_unet_model.pth`.
- **Test set evaluation** reports final loss, Dice, and Jaccard scores.
- **Qualitative results**: Visualizations show original images, ground truth masks, predicted masks, and blended overlays (red: prediction, green: ground truth, yellow: intersection).

- ![image](https://github.com/user-attachments/assets/229a40de-cf62-415a-b82d-fe683d6d16bc)



## Requirements

- Python 3.7+
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- scikit-learn
- pillow
- gdown

Install dependencies with:

```sh
pip install torch torchvision numpy pandas matplotlib scikit-learn pillow gdown
```

## References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [VGG Networks](https://arxiv.org/abs/1409.1556)
