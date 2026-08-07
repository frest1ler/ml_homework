# Neural Network From Scratch vs PyTorch

> Сравнение собственной реализации полносвязной нейронной сети на NumPy с реализацией на PyTorch на задаче классификации изображений MNIST.

---

## Description

Проект демонстрирует, как устроены основные компоненты современной нейронной сети "под капотом".

В рамках работы реализована собственная библиотека для построения и обучения нейронных сетей, после чего её качество и производительность сравниваются с аналогичной моделью, реализованной на **PyTorch**.

Проект предназначен для изучения принципов работы глубокого обучения и понимания внутренних механизмов современных DL-фреймворков.

---

## Features

* ✔ Собственная реализация нейронной сети без использования DL-фреймворков
* ✔ Полносвязные (`Linear`) слои
* ✔ Реализация прямого и обратного распространения ошибки (Forward / Backward propagation)
* ✔ Batch Normalization
* ✔ Dropout
* ✔ Несколько функций активации:

  * ReLU
  * LeakyReLU
  * ELU
  * SoftPlus
* ✔ Реализация функций потерь (Loss)
* ✔ Собственные оптимизаторы:

  * SGD + Momentum
  * Adam
* ✔ Подбор гиперпараметров с помощью Optuna
* ✔ Сравнение результатов с реализацией на PyTorch
* ✔ Обучение на датасете MNIST

---

# Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Optimization-blueviolet)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter)

---

# Getting Started

## Prerequisites

Перед запуском убедитесь, что установлены:

* Python 3.10+
* pip
* git
* venv

Проверить можно командами

```bash
python3 --version
pip3 --version
```

---

## Installation

Клонировать репозиторий

```bash
git clone <repository-url>

cd <repository-name>
```

Создать виртуальное окружение

```bash
python3 -m venv .venv
```

Активировать его

```bash
source .venv/bin/activate
```

Установить зависимости

```bash
pip install -U pip

pip install numpy matplotlib scipy scikit-learn jupyter notebook torch torchvision optuna tqdm
```

---

## Environment

Проект не требует обязательного файла `.env`.

При необходимости можно создать файл

```text
.env
```

например

```env
PYTHONUNBUFFERED=1
```

---

## Run

Запустить Jupyter Notebook

```bash
jupyter notebook
```

или

```bash
jupyter lab
```

Далее открыть

```
main_notebook.ipynb
```

и выполнить все ячейки последовательно.

---

# Project Structure

```
.
├── main_notebook.ipynb      # основной ноутбук обучения и сравнения моделей
├── modules.ipynb            # реализация собственной библиотеки нейронных сетей
├── mnist.py                 # загрузка датасета MNIST
├── test.ipynb               # эксперименты
└── README.md
```

---

# Самая интересная часть проекта

Ключевой особенностью проекта является самостоятельная реализация практически всех базовых компонентов современной нейронной сети без использования готовых инструментов глубокого обучения.

В проекте реализованы:

* вычислительный граф;
* механизм обратного распространения ошибки;
* собственные слои;
* Batch Normalization;
* Dropout;
* оптимизаторы SGD и Adam.

Дополнительно выполняется автоматический подбор гиперпараметров с помощью **Optuna**, после чего результаты сравниваются с эквивалентной моделью, построенной на **PyTorch**.

Такой подход позволяет не только получить рабочую модель классификации MNIST, но и глубже понять внутреннее устройство современных библиотек глубокого обучения.

---

## Результат

Проект показывает, что собственная реализация нейронной сети способна успешно решать задачу классификации изображений MNIST, однако использование PyTorch значительно упрощает разработку, ускоряет обучение моделей и предоставляет готовые высокооптимизированные инструменты для построения современных архитектур глубокого обучения.
