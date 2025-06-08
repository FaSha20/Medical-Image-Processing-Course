# VoxelMorph

## Overview
This Jupyter Notebook implements the **VoxelMorph** framework, a deep learning approach for unsupervised image registration, as described in the paper [VoxelMorph: A Learning Framework for Deformable Medical Image Registration](https://arxiv.org/pdf/1809.05231). 

The project focuses on aligning medical images into a common coordinate system using a convolutional neural network (CNN) to predict a deformation field. This deformation field maps pixels from one image to another, enabling accurate spatial correspondence.

## Contents
1. **Introduction**: Describes the VoxelMorph framework and its purpose in image registration.
2. **Dependencies**: Lists required Python libraries and tools.
3. **Data Loading**: Implements a data loader using the **MedNISTDataset** for medical image registration.
4. **Model Implementation**: Defines the VoxelMorph CNN architecture (incomplete in the provided snippet).
5. **Visualization**: Includes code to display sample images (output shown as a matplotlib figure).

## Requirements
To run the notebook, ensure you have the following dependencies installed:

### Python Version
- Python 3.8.10 (or compatible)

### Libraries
Install the required libraries using `pip`:
```bash
pip install numpy matplotlib scikit-image torch torchvision monai tqdm
```

Key libraries used:
- **NumPy**: For numerical computations.
- **Matplotlib**: For plotting and visualization.
- **Scikit-Image**: For image processing.
- **PyTorch**: For building and training the CNN.
- **MONAI**: For medical image processing and data loading.
- **tqdm**: For progress bars during training.

### Hardware
- **GPU** (recommended): The notebook is configured to use a GPU (e.g., T4 on Google Colab) for faster training.
- **CPU**: Can be used, but training may be slower.

## Setup Instructions
1. **Clone or Download the Notebook**:
   - Download `VM_IABI_Prac_hw4.ipynb` to your local machine or Google Colab.

2. **Install Dependencies**:
   - Run the following command in your environment to install MONAI and other dependencies:
     ```bash
     pip install monai
     ```
   - The notebook includes a cell that installs MONAI automatically:
     ```python
     !pip install monai
     ```

3. **Prepare the Dataset**:
   - The notebook uses the **MedNISTDataset**, which is automatically downloaded by MONAI when the code is executed.
   - The dataset is stored in a temporary directory (`/tmp/tmpsznb9o4t` in the provided output).
   - Ensure you have internet access to download the dataset (`MedNIST.tar.gz`, ~59 MB).

4. **Run the Notebook**:
   - Open the notebook in Jupyter or Google Colab.
   - Execute the cells sequentially to:
     - Install dependencies.
     - Load and preprocess the MedNIST dataset.
     - Visualize sample images.
   - Note: The provided notebook snippet is incomplete, so additional cells (e.g., model definition, training loop) may need to be implemented to fully reproduce the VoxelMorph framework.

## Dataset
The **MedNISTDataset** is used for this project, containing medical images (e.g., hand X-rays). Key details:
- **Source**: Automatically downloaded via MONAI.
- **Structure**: Images are stored in a directory structure (e.g., `/tmp/tmpsznb9o4t/MedNIST/Hand/`).
- **Preprocessing**:
  - Images are resized to 64x64 pixels.
  - Transformations include:
    - `EnsureChannelFirstD`: Ensures channel-first format.
    - `LoadImageD`: Loads images from disk.
    - `RandRotateD`, `RandZoomD`: Applies random rotations and zooms for data augmentation.
    - `ScaleIntensityRanged`: Normalizes image intensity.

Sample data loader output:
```python
first training items: [
    {'fixed_hand': '/tmp/tmpsznb9o4t/MedNIST/Hand/003836.jpeg', 'moving_hand': '/tmp/tmpsznb9o4t/MedNIST/Hand/003836.jpeg'},
    {'fixed_hand': '/tmp/tmpsznb9o4t/MedNIST/Hand/001613.jpeg', 'moving_hand': '/tmp/tmpsznb9o4t/MedNIST/Hand/001613.jpeg'},
    ...
]
```

## Model Description
The VoxelMorph framework uses a CNN to predict a deformation field for image registration. The model:
- Takes a pair of images (fixed and moving) as input.
- Outputs a deformation field that warps the moving image to align with the fixed image.
- Is trained in an unsupervised manner, minimizing a loss function based on image similarity and deformation smoothness.

**Note**: The provided notebook snippet does not include the model definition or training loop. Refer to the VoxelMorph paper or official repository for implementation details:
- [VoxelMorph GitHub](https://github.com/voxelmorph/voxelmorph)

## Usage
To use the notebook:
1. **Execute the Dependency Cell**:
   - Run the cell that imports libraries and installs MONAI.
2. **Run the Data Loader Cell**:
   - This downloads the MedNIST dataset, applies transformations, and visualizes sample images.
3. **Implement the Model** (if extending the notebook):
   - Define the VoxelMorph CNN architecture (e.g., U-Net-based).
   - Implement the loss function (e.g., mean squared error for image similarity + regularization for deformation smoothness).
   - Set up the training loop using PyTorch.
4. **Train and Evaluate**:
   - Train the model on the MedNIST dataset.
   - Evaluate registration accuracy using metrics like Dice coefficient or normalized cross-correlation.

## Output
The provided notebook generates a visualization of sample images from the MedNIST dataset, displayed as a matplotlib figure with 40 subplots. Each subplot shows a pair of fixed and moving images (shape: 64x64 pixels).

Example output:
- **Console**:
  ```
  first training items: [{'fixed_hand': '.../003836.jpeg', 'moving_hand': '.../003836.jpeg'}, ...]
  moving_image shape: torch.Size([64, 64])
  fixed_image shape: torch.Size([64, 64])
  ```
- **Visualization**: A figure displaying multiple image pairs (included in the notebook output).

## References
- **VoxelMorph Paper**: Balakrishnan, G., Zhao, A., Sabuncu, M. R., Guttag, J., & Dalca, A. V. (2018). [VoxelMorph: A Learning Framework for Deformable Medical Image Registration](https://arxiv.org/pdf/1809.05231).
- **MONAI Documentation**: [https://docs.monai.io](https://docs.monai.io)
- **PyTorch Documentation**: [https://pytorch.org/docs](https://pytorch.org/docs)
- **MedNIST Dataset**: Provided by MONAI.
