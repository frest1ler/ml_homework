# SVM with Kernel Methods

## Overview

This project implements a Support Vector Machine (SVM) classifier from
scratch using PyTorch, with support for kernel methods such as: - Linear
kernel - RBF (Gaussian) kernel

------------------------------------------------------------------------

## Features

-   Custom SVM implementation\
-   Kernel support\
-   Mini-batch training\
-   Hinge loss optimization

------------------------------------------------------------------------

## Project Structure

. ├── svm.py\
├── assignment0_03_svm_kernel.ipynb\
└── README.md

------------------------------------------------------------------------

## Installation

``` bash
python -m venv .venv
source .venv/bin/activate
pip install numpy torch scikit-learn matplotlib seaborn
```

------------------------------------------------------------------------

## Usage Example

``` python
from svm import SVM, rbf
import numpy as np

X = np.random.randn(100, 2)
y = np.random.choice([-1, 1], size=100)

model = SVM(lr=1e-3, epochs=10, batch_size=32, kernel_function=rbf)
model.fit(X, y)

preds = model.predict(X)
```
