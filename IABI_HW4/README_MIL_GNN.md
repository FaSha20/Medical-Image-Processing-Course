# Multiple Instance Learning with Graph Neural Networks

### IABI Course – Dr. Rohban  
**Homework 1**  
**Student:** Fatemeh Shahhosseini  
**SID:** 403206519

## 📘 Project Overview

This project explores **Multiple Instance Learning (MIL)** using **Graph Neural Networks (GNNs)** to classify bags of instances. Each bag is treated as a graph where:
- **Nodes** represent individual instances (images),
- **Edges** encode similarities or relationships between them.

The learning process involves generating meaningful **graph-level embeddings** that are later used for **bag-level classification**.

## 🧠 Dataset

We use the **MNIST Bags** dataset. Each "bag" contains several digit images, and:
- A bag is labeled **positive** if it contains at least one digit '9'.
- Otherwise, it is labeled **negative**.

## 🔍 Methodology

- Each bag is converted into a **graph structure** based on visual similarity between digits.
- Graphs are passed through a **GNN architecture** to extract global features.
- Classification is performed using the **graph-level embedding**.
- Model performance is evaluated using standard metrics such as **accuracy**, **precision**, and **recall**.

## 🛠️ Key Libraries Used

```python
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch_geometric.data import Data, DataLoader, NeighborSampler
from chamferdist import ChamferDistance
```

## 📊 Results

The notebook evaluates the performance of the GNN-based MIL model using accuracy and other classification metrics. The aim is to demonstrate the effectiveness of incorporating structure (graph) into MIL settings.

## 🧪 How to Run

1. Clone this repository or download the notebook.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure you have `torch`, `torch_geometric`, and `chamferdist` installed.
4. Run the notebook:
   ```bash
   jupyter notebook IABI_Prac_hw4_MIL_Fatemeh_Shahhosseini_403206519.ipynb
   ```

---

## 📂 File Structure

- `IABI_Prac_hw4_MIL_Fatemeh_Shahhosseini_403206519.ipynb`: Main notebook with implementation.
- `README.md`: This documentation.
