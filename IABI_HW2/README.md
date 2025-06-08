# Brain Tumor Detection Using CNN (PyTorch)

## 📚 Project Overview

This project is a part of the **Intelligent Analysis of Biomedical Images** course at Sharif University of Technology, CE Department, under the supervision of **Dr. Rohban**.

> **Goal**: Develop a Convolutional Neural Network (CNN) using the PyTorch framework to accurately detect and classify brain tumors from MRI scans.

## 🧠 Dataset

The model is trained on a large dataset of labeled MRI brain scan images, including:
- **Tumor** images
- **Healthy** images

## 🚀 Key Features

- Image preprocessing and visualization
- Custom PyTorch Dataset class
- CNN architecture design and training loop
- Evaluation using accuracy and classification reports
- Learning rate scheduling and model optimization techniques

## 🛠️ Libraries Used

The notebook relies on the following main libraries:

```python
from PIL import Image
from sklearn.metrics import classification_report
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, random_split
from torchsummary import summary
from torchvision import utils
```

## 📊 Results

The final model is evaluated using standard classification metrics, such as precision, recall, and F1-score, to validate its effectiveness in distinguishing between healthy and tumorous MRI scans.

## 📝 How to Run

1. Clone the repository.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Open and run the notebook:
   ```bash
   jupyter notebook Brain_Tumor.ipynb
   ```
