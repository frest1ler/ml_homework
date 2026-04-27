# Decision Tree from Scratch

## Overview

This project implements a **Decision Tree** classifier and regressor from scratch using **NumPy**, with support for multiple splitting criteria and configurable tree depth. The implementation follows the scikit-learn interface, making it easy to integrate into existing ML pipelines.

------------------------------------------------------------------------

## Features

-   **Classification & Regression** support
-   **Splitting criteria**:
    -   Gini impurity
    -   Entropy (information gain)
    -   Variance (for regression)
    -   Mean absolute deviation from the median (MAD)
-   Recursive tree building with early stopping
-   `predict` and `predict_proba` methods
-   Fully compatible with `sklearn.BaseEstimator`

------------------------------------------------------------------------

## Project Structure
tree.py                            # Core DecisionTree implementation
assignment0_04_decision_tree.ipynb # Experiments and usage examples
README.md

------------------------------------------------------------------------

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate
pip install numpy scikit-learn matplotlib

------------------------------------------------------------------------

Usage Example

``` python
import numpy as np
from tree import DecisionTree

# Classification data
X = np.random.randn(100, 2)
y = np.random.choice([0, 1], size=(100, 1))

model = DecisionTree(max_depth=5, criterion_name='gini')
model.fit(X, y)

preds = model.predict(X)
probs = model.predict_proba(X)

# Regression data
y_reg = np.random.randn(100, 1)
regressor = DecisionTree(max_depth=3, criterion_name='variance')
regressor.fit(X, y_reg)
reg_preds = regressor.predict(X)
```
