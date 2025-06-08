# Image Enhancement and Pyramid Processing

This notebook ([signal.ipynb](signal.ipynb)) demonstrates several fundamental image processing techniques using OpenCV and NumPy, focusing on frequency analysis, image pyramids, unsharp masking, and global enhancement. The workflow is designed for educational purposes, providing step-by-step code and visualizations for each method.

---

## Table of Contents

- [Overview](#overview)
- [Notebook Structure](#notebook-structure)
- [How to Run](#how-to-run)
- [Key Components](#key-components)
- [Requirements](#requirements)
- [References](#references)

---

## Overview

The notebook covers the following topics:

- **Image Loading and Preprocessing:** Reads an image, converts it to grayscale, and resizes it.
- **Frequency Domain Analysis:** Computes and visualizes the 2D FFT magnitude spectrum.
- **Image Pyramid Construction:** Builds a detail pyramid using Gaussian downsampling and upsampling.
- **Pyramid Reconstruction:** Reconstructs the original image from the pyramid layers.
- **Unsharp Masking:** Enhances image sharpness using both Gaussian blur and pyramid-based methods.
- **Global Enhancement:** Applies a non-linear (tanh-based) global contrast enhancement to pyramid layers.
- **Visualization:** Displays results at each step for comparison and analysis.

---

## Notebook Structure

1. **Imports and Setup:**  
   Loads OpenCV, NumPy, and Matplotlib.

2. **Primary Image:**  
   Loads and preprocesses the input image (`download.jfif`), converts to grayscale, and resizes to 128x128.

3. **Frequency Domain Analysis:**  
   - Computes the 2D FFT of the image.
   - Visualizes the log-magnitude spectrum and a normalized version.

4. **Image Detail Pyramid:**  
   - Defines `image_pyramid` to decompose the image into multiple detail layers using Gaussian pyramids.
   - Visualizes each detail layer and the original image.

5. **Pyramid Reconstruction:**  
   - Defines `image_pyrm_rec` to reconstruct the image from its pyramid layers.
   - Visualizes partial and full reconstructions.

6. **Unsharp Masking:**  
   - Computes sharp edges using both Gaussian blur and pyramid-based methods.
   - Visualizes the original, edge, and sharpened images.

7. **Global Enhancement:**  
   - Defines `global_enhancement` using a tanh-based nonlinearity.
   - Enhances each pyramid layer with different slopes.
   - Visualizes enhanced layers and reconstructs the enhanced image.

---

## How to Run

1. Place your input image as `download.jfif` in the notebook's working directory.
2. Open [signal.ipynb](signal.ipynb) in Jupyter Notebook or VS Code.
3. Run all cells in order to reproduce the results and visualizations.

---

## Key Components

- **image_pyramid(I, n_levels):**  
  Decomposes an image into detail layers using Gaussian downsampling and upsampling.

- **image_pyrm_rec(detail_pyrm, n_level):**  
  Reconstructs the image from its detail pyramid.

- **global_enhancement(I, slope):**  
  Applies a tanh-based global contrast enhancement to an image or image layer.

- **Unsharp Masking:**  
  Enhances image sharpness by subtracting a blurred version from the original.

---

## Requirements

- Python 3.7+
- numpy
- opencv-python
- matplotlib

Install dependencies with:

```sh
pip install numpy opencv-python matplotlib
```

---

## References

- [OpenCV Documentation](https://docs.opencv.org/)
- [NumPy FFT](https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html)
- [Image Pyramids in OpenCV](https://docs.opencv.org/4.x/dc/dff/tutorial_py_pyramids.html)
- [Unsharp Masking](https://en.wikipedia.org/wiki/Unsharp_masking)
