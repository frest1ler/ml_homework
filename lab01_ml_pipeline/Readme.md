# lab1_ml_pipeline

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Jupyter](https://img.shields.io/badge/jupyter-notebook-orange)

## Overview

**lab1_ml_pipeline** is a machine learning laboratory project focused on data preprocessing, model training, dimensionality reduction, and classification using Logistic Regression, PCA, and Support Vector Machines (SVM). The project demonstrates a complete machine learning workflow, from feature scaling and model evaluation to kernel-based classification and feature engineering.

---

## Key Features

* Data preprocessing and feature scaling
* Logistic Regression training and hyperparameter tuning
* Model evaluation using Accuracy and F1-score
* ROC curve visualization for multiclass classification
* Principal Component Analysis (PCA)
* Support Vector Machine (SVM) classification
* Kernel-based learning (`linear`, `poly`, `rbf`, `sigmoid`)
* Feature engineering with polynomial features
* Decision boundary visualization

---

## Requirements

### Python

* Python 3.9+

### Main Dependencies

| Package      | Purpose                     |
| ------------ | --------------------------- |
| numpy        | Numerical computations      |
| pandas       | Data manipulation           |
| matplotlib   | Visualization               |
| scikit-learn | Machine learning algorithms |
| mlxtend      | Decision region plotting    |
| jupyter      | Notebook environment        |

### Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn mlxtend jupyter
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<username>/lab1_ml_pipeline.git
cd lab1_ml_pipeline
```

### 2. Create a Virtual Environment

#### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

or

```bash
pip install numpy pandas matplotlib scikit-learn mlxtend jupyter
```

---

## Project Structure

```text
lab1_ml_pipeline/
│
├── Lab1_part1_questions.ipynb
├── Lab1_part2_ml_pipeline.ipynb
├── Lab1_part3_SVM.ipynb
│
├── car_data.csv
│
├── requirements.txt
└── README.md
```

### Module Description

| File                           | Description                                                    |
| ------------------------------ | -------------------------------------------------------------- |
| `Lab1_part1_questions.ipynb`   | Matrix differentiation and optimization exercises              |
| `Lab1_part2_ml_pipeline.ipynb` | Data preprocessing, Logistic Regression, PCA, model evaluation |
| `Lab1_part3_SVM.ipynb`         | SVM experiments, kernels, feature generation, decision regions |
| `car_data.csv`                 | Vehicle silhouette classification dataset                      |

---

## Dataset

Main tasks:

1. Data preprocessing
2. Feature scaling
3. Logistic Regression
4. Hyperparameter optimization
5. PCA analysis
6. SVM classification
7. Kernel comparison
8. Feature engineering with polynomial features
9. Performance evaluation and visualization

---
