# 🧠 Task 4: Artificial Neural Network from Scratch on MNIST

This task involves implementing an Artificial Neural Network (ANN) **completely from scratch**, without using high-level libraries like TensorFlow or PyTorch. You will train the network on the **MNIST handwritten digit dataset** using **gradient descent** and implement all the key components manually.

---

## 📌 Objective

- Build an ANN from scratch
- Understand and implement:
  - Forward propagation
  - Backpropagation
  - Loss functions (e.g., Cross-Entropy)
  - Gradient Descent Optimizer
  - Activation functions (ReLU, Sigmoid, etc.)
- Train on the **MNIST dataset**
- Evaluate performance on unseen test data

---

## 📚 Resources

- [3Blue1Brown – Deep Learning, Visualized](https://youtu.be/aircAruvnKk?si=E7aowG9bgxeQ-CoI)

Highly recommended before beginning — this video explains the "why" behind each concept you’ll implement.

---

## 🗂️ Files Included

| File Name                 | Description                                      |
|--------------------------|--------------------------------------------------|
| `ann_from_scratch.py`    | Python script implementing ANN manually          |
| `mnist_loader.py`        | Utility to download/load MNIST data (optional)  |
| `README.md`              | This file                                       |
| `report.pdf` (optional)  | Brief summary of approach, accuracy, and insights|

---

## 📊 Dataset: MNIST

The **MNIST** dataset contains 70,000 grayscale images (28x28) of handwritten digits (0–9).

You can download it using:
```python
from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
