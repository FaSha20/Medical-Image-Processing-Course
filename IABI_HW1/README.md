# Retinal Blood Vessels Segmentation

## Overview

This notebook implements a **retinal blood vessel segmentation** pipeline using classical image processing and a simple neural network classifier. The goal is to classify each pixel in retinal images from the DRIVE dataset as either vessel or background, based on a rich set of handcrafted features.

---

## Table of Contents

- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Preprocessing](#preprocessing)
  - [Feature Extraction](#feature-extraction)
  - [Data Preparation](#data-preparation)
  - [Model Architecture](#model-architecture)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Results](#results)
- [Requirements](#requirements)
- [References](#references)

---

## Dataset

- **DRIVE**: Digital Retinal Images for Vessel Extraction
- Contains 20 training and 20 test images, each with a corresponding mask and manual vessel annotation.
- Labels: Binary images (255 = vessel, 0 = background)
- [DRIVE dataset info](https://drive.grand-challenge.org/)

---

## Methodology

### Preprocessing

- **Channel Selection:** The green channel is used for vessel segmentation, as it provides the best contrast and detail.
- **Masking:** Non-retinal regions are replaced with the mean intensity of the masked area.
- **Morphological Opening:** Removes small noise.
- **CLAHE:** Enhances local contrast.
- **Median Filtering:** Smooths the image.

### Feature Extraction

For each pixel (within the mask), a **20-dimensional feature vector** is computed, including:

1. **Edge Features:**  
   - Roberts, Prewitt, Sobel, Canny, and Laplacian of Gaussian (LoG) edge detectors.

2. **Morphological Features:**  
   - Top-hat and black-hat transforms with different structuring elements.

3. **Gradient-Based Features:**  
   - Horizontal and vertical gradients, magnitude, and orientation.

4. **Hessian Features:**  
   - Second-order derivatives (Gxx, Gxy, Gyy).

5. **Statistical Features (in a 21x21 patch):**  
   - Mean, min, max, skewness, kurtosis, standard deviation, mean absolute deviation, root sum of squares.

### Data Preparation

- Features and labels are extracted for all pixels inside the mask for each image.
- Data is saved as CSV files for both training and test sets.
- Features are standardized using `StandardScaler`.

### Model Architecture

- **Classifier:** A simple feedforward neural network with three hidden layers (20→32→32→16→1), using Tanh activations and a sigmoid output.
- **Loss:** Weighted binary cross-entropy to address class imbalance (vessel pixels are much fewer than background).

### Training

- Trained for 20 epochs using SGD.
- Batch size: 128
- Learning rate: 0.01
- Training and validation accuracy and loss are tracked.

### Evaluation

- **Metrics:** Sensitivity, specificity, and accuracy, computed as:
  - Sensitivity = TP / (TP + FN)
  - Specificity = TN / (TN + FP)
  - Accuracy = (TP + TN) / (TP + TN + FP + FN)
- **Threshold:** Output probabilities are binarized at 0.4.
- **Visualization:** For two test images, the original, ground truth, and model output are displayed side by side.

---

## Results

- The model achieves at least **70% sensitivity**, **90% specificity**, and **90% accuracy** on the test set (as required).
- Example output (for two test images):

| Original Image | Ground Truth | Model Output |
|:--------------:|:------------:|:------------:|
| ![Sample1](sample1.png) | ![GT1](gt1.png) | ![Pred1](pred1.png) |
| ![Sample2](sample2.png) | ![GT2](gt2.png) | ![Pred2](pred2.png) |

> *(Replace the above image links with your actual saved images if you want to display results in your repository.)*

---

## Requirements

- Python 3.7+
- numpy
- pandas
- opencv-python
- scikit-image
- scipy
- matplotlib
- tqdm
- torch
- scikit-learn
- imageio
- gdown

Install dependencies with:

```sh
pip install numpy pandas opencv-python scikit-image scipy matplotlib tqdm torch scikit-learn imageio gdown
```

---

## References

- [DRIVE: Digital Retinal Images for Vessel Extraction](https://drive.grand-challenge.org/)
- [Feature Extraction Techniques in Image Processing](https://scikit-image.org/)
- [PyTorch Documentation](https://pytorch.org/)

