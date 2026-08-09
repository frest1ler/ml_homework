# 🧠 Lab 02 — Deep Learning

> Исследовательская лабораторная работа по Deep Learning: от базовых fully-connected сетей и работы с overfitting до генерации текста с RNN и классификации изображений с использованием CNN и fine-tuning предобученных моделей.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Torchvision](https://img.shields.io/badge/Torchvision-Computer%20Vision-EE4C2C?logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

---

## 📌 О проекте

**Lab 02 — Deep Learning** — набор экспериментов, посвящённых различным классам нейронных сетей и практическим методам обучения моделей.

Работа последовательно охватывает:

- fully-connected networks и исследование **overfitting**;
- регуляризацию с помощью **Dropout** и **Batch Normalization**;
- генерацию текста с помощью **Vanilla RNN**;
- обработку русскоязычного текста на примере **«Евгения Онегина»**;
- исследование задач Computer Vision и CNN;
- классификацию изображений;
- **Transfer Learning / Fine-Tuning** предобученных моделей из `torchvision`;
- сравнение `ResNet50`, `EfficientNet-B0`, `MobileNetV3-Large` и `ViT-B/16`;
- двухфазное обучение: сначала classifier head, затем разморозка backbone;
- сохранение лучших весов и истории обучения;
- сравнение результатов нескольких архитектур в единой таблице.

Главная идея проекта — не просто обучить одну модель, а исследовать, **как выбор архитектуры, регуляризации, стратегии обучения и предварительно обученных весов влияет на результат**.

---

## ✨ Основные возможности

- ✅ Обучение fully-connected нейронных сетей на Fashion-MNIST
- ✅ Экспериментальное создание и анализ overfitting
- ✅ Использование Dropout и Batch Normalization
- ✅ Построение character-level Vanilla RNN
- ✅ Генерация текста на русском языке
- ✅ Работа с «Евгением Онегиным»
- ✅ Работа с Computer Vision и CNN
- ✅ Классификация изображений
- ✅ Transfer Learning с моделями `torchvision`
- ✅ Fine-Tuning backbone предобученных моделей
- ✅ Двухфазная стратегия обучения
- ✅ Поддержка SGD, Adam и AdamW
- ✅ `CosineAnnealingLR` и `StepLR`
- ✅ Data Augmentation для изображений
- ✅ Автоматическое определение `CUDA` / `CPU`
- ✅ Benchmark нескольких архитектур в одном pipeline
- ✅ Сохранение лучших `.pth` весов
- ✅ Сохранение истории обучения
- ✅ Сводная таблица результатов
- ✅ Визуализация `loss` и `accuracy`
- ⚙️ Конфигурация экспериментов и search spaces для настройки гиперпараметров

---

## 🧰 Tech Stack

| Технология | Назначение |
|---|---|
| **Python** | Основной язык проекта |
| **PyTorch** | Обучение нейронных сетей |
| **Torchvision** | Датасеты, transforms и pretrained CV-модели |
| **NumPy** | Работа с массивами и численными данными |
| **Pandas** | Анализ и сравнение результатов |
| **Matplotlib** | Графики loss/accuracy и визуализация |
| **scikit-learn** | Разбиение данных и метрики |
| **tqdm** | Progress bars при обучении |
| **Jupyter Notebook** | Основной формат экспериментов |

---

# 📚 Содержание

- [Структура лабораторной](#-структура-лабораторной)
- [Результаты](#-результаты)
- [Установка](#-установка)
- [Запуск](#-запуск)
- [Part 2 — Overfitting](#part-2--overfitting)
- [Part 3 — Poetry Generation](#part-3--poetry-generation)
- [Part 4 — Computer Vision](#part-4--computer-vision)
- [Part 5 — Fine-Tuning](#part-5--fine-tuning)
- [Предобученные модели](#-предобученные-модели)
- [Тестирование](#-тестирование)
- [Contributing](#-contributing)

---

# 🗂 Структура лабораторной

| Часть | Тема | Основная задача |
|---|---|---|
| **Part 2** | Overfitting | Fully-connected network, анализ переобучения и регуляризация |
| **Part 3** | Poetry Generation | Character-level Vanilla RNN и генерация текста |
| **Part 4** | Computer Vision | Работа с изображениями и CNN |
| **Part 5** | Dogs Classification | Transfer Learning и Fine-Tuning pretrained моделей |

---

# 📊 Результаты

## 🐶 Image Classification / Fine-Tuning

В `Part 5` реализован единый pipeline для сравнения нескольких pretrained моделей.

Модели:

- `ResNet50`
- `EfficientNet-B0`
- `MobileNetV3-Large`
- `ViT-B/16`

В ноутбуке используется разбиение исходной выборки на:

- **5732** training images;
- **1434** validation images.

Для обучения используются аугментации:

- `RandomResizedCrop(224)`;
- `RandomHorizontalFlip`;
- `ColorJitter`;
- `Normalize` с ImageNet mean/std.

Для validation используется resize/crop без случайных аугментаций.

### Лучшие результаты

| Model | Best Validation Accuracy | Last Train Accuracy | Last Validation Accuracy |
|---|---:|---:|---:|
| 🥇 **ViT-B/16** | **98.68%** | 95.80% | 98.47% |
| 🥈 **ResNet50** | **96.23%** | 88.40% | 96.09% |
| 🥉 **EfficientNet-B0** | **91.77%** | 79.55% | 91.63% |
| **MobileNetV3-Large** | **90.79%** | 84.18% | 90.66% |

> Значения выше взяты из итоговой таблицы `results_df` в `Lab2_DL_parts_5.ipynb`.

---

## 🎯 Отдельно зафиксированные результаты

В экспериментах также получены следующие значения:

- `ViT-B/16` — до **98.68%** best validation accuracy;
- `ResNet50` — до **97.00%** best validation accuracy в отдельном эксперименте;
- `MobileNetV3-Large` — около **90.03%** validation accuracy в одном из checkpoints.

Разные значения могут соответствовать разным конфигурациям/моментам обучения, поэтому для сравнения архитектур рекомендуется использовать итоговую таблицу `results_df`.

---

# 🔬 Что именно исследуется

## 1. Fully-Connected Networks

В `Part 2` используется Fashion-MNIST.

Базовая задача — получить не менее `88.5%` test accuracy на fully-connected модели.

В качестве компонентов сети используются:

- `Linear`;
- `ReLU`;
- `BatchNorm1d`;
- `Dropout`.

В одном из экспериментов модель достигла:

```text
Val Acc: 0.8898
```

Также построена более глубокая модель для намеренного создания overfitting.

---

## 2. Overfitting

Отдельный эксперимент показывает поведение модели при увеличении её сложности.

Для overfitting-модели используются несколько последовательных `Linear` слоёв:

```text
784
 ↓
392
 ↓
196
 ↓
98
 ↓
49
 ↓
10
```

При длительном обучении training loss продолжает уменьшаться, в то время как validation loss начинает расти.

Это позволяет на практике увидеть различие между:

- способностью модели запоминать training set;
- обобщающей способностью на validation set.

---

## 3. Regularization

Для борьбы с overfitting исследуются:

- Dropout;
- Batch Normalization;
- изменение архитектуры;
- контроль сложности модели.

Таким образом, Part 2 является не только задачей классификации, но и небольшим исследованием поведения нейронной сети при изменении её capacity.

---

# ✍️ Part 3 — Poetry Generation

В Part 3 реализована **character-level генерация текста** с использованием Vanilla RNN.

В качестве источника текста рассматривается:

> Александр Сергеевич Пушкин — «Евгений Онегин»

Также в исходном задании предусмотрен вариант с Shakespeare Sonnets.

## Pipeline

Обучение проходит через несколько этапов:

```text
Raw text
   ↓
Preprocessing
   ↓
Vocabulary
   ↓
Token → Index
   ↓
Sequence Dataset
   ↓
Embedding
   ↓
Vanilla RNN
   ↓
Linear
   ↓
Character prediction
   ↓
Text generation
```

Для «Евгения Онегина» используется алфавит из **38 токенов**:

```text
! , . ? _ а б в г д е ё ж з и й к л м н о
п р с т у ф х ц ч ш щ ъ ы ь э ю я
```

Пробел заменяется на `_`, после чего текст фильтруется по допустимым символам.

### Архитектура RNN

```text
Embedding
    ↓
RNN
    ↓
Linear
    ↓
Vocabulary logits
```

Параметры эксперимента:

```text
SEQ_LEN   = 100
BATCH_SIZE = 64
embedding size = 64
hidden size    = 128
optimizer      = Adam
learning rate  = 0.001
epochs         = 20
```

Финальная loss в приведённом эксперименте находится около:

```text
1.726
```

---

# 👁 Part 4 — Computer Vision

В Part 4 рассматриваются задачи Computer Vision и классификации.

В ноутбуке исследуется работа с CNN/ResNet-подобной архитектурой. В частности, присутствуют стандартные компоненты сверточных сетей:

- `Conv2d`;
- `BatchNorm2d`;
- `ReLU`;
- `MaxPool2d`;
- residual/bottleneck blocks.

Экспериментальная часть ориентирована на подбор архитектуры и получение accuracy не менее `86%`, с более высокими целями в зависимости от качества решения.

---

# 🐕 Part 5 — Fine-Tuning

Part 5 — наиболее развитая с инженерной точки зрения часть лабораторной.

Задача — классификация **50 пород собак**.

Вместо обучения большой модели полностью с нуля используется **Transfer Learning**.

Предобученная модель загружается с ImageNet weights, после чего её classifier заменяется на новый, рассчитанный на количество классов целевого датасета.

---

## 🏗 Архитектуры

Pipeline поддерживает:

```text
ResNet50
EfficientNet-B0
MobileNetV3-Large
ViT-B/16
```

Архитектуры создаются через конфигурацию:

```python
MODEL_CONFIGS = {
    "resnet50": ...,
    "efficientnet_b0": ...,
    "mobilenet_v3_large": ...,
    "vit_b_16": ...
}
```

Это позволяет использовать один training pipeline для разных моделей.

---

# 🔥 Двухфазный Fine-Tuning

Основная особенность Part 5 — двухэтапная стратегия обучения.

## Phase 1 — classifier training

Сначала backbone замораживается:

```python
for param in model.parameters():
    param.requires_grad = False
```

После этого обучается только classifier.

Схематично:

```text
Pretrained Backbone
       │
       │ frozen
       ▼
New Classifier
       │
       ▼
50 classes
```

---

## Phase 2 — Fine-Tuning

После обучения classifier backbone размораживается:

```python
for param in model.parameters():
    param.requires_grad = True
```

Затем вся сеть дообучается с меньшим learning rate.

```text
Pretrained Backbone
       │
       │ trainable
       ▼
New Classifier
       │
       ▼
50 classes
```

Такой подход позволяет сначала адаптировать новый classifier, а затем аккуратно подстроить pretrained feature extractor под целевой датасет.

---

# ⚙️ Training Pipeline

В Part 5 реализованы отдельные функции для основных этапов:

```text
get_model()
      ↓
get_train_config()
      ↓
build_dataloaders()
      ↓
build_optimizer()
      ↓
build_scheduler()
      ↓
train_epoch()
      ↓
validate()
      ↓
train_phase()
      ↓
train_model()
      ↓
train_all_models()
      ↓
compare_models()
```

Pipeline умеет:

- создавать модели;
- заменять classifier;
- строить DataLoader;
- замораживать backbone;
- размораживать backbone;
- создавать optimizer;
- создавать scheduler;
- запускать training phase;
- сохранять лучший checkpoint;
- сохранять history;
- сравнивать несколько моделей.

---

# 🧪 Optimizers & Schedulers

В проекте используются:

### Optimizers

- SGD
- Adam
- AdamW

### Learning Rate Schedulers

- `CosineAnnealingLR`
- `StepLR`

Для разных моделей предусмотрены разные конфигурации обучения.

Например, для ResNet50 в конфигурации используются SGD + momentum:

```python
SGD(
    lr=1e-2,
    momentum=0.9,
    weight_decay=1e-4
)
```

А для ViT-B/16 используется AdamW с небольшим learning rate:

```python
AdamW(
    lr=3e-5,
    weight_decay=5e-2
)
```

---

# 🎛 Hyperparameter Tuning

В конфигурации моделей предусмотрены отдельные пространства параметров для настройки:

```text
optimizer
learning rate
batch size
weight decay
```

Например, для ResNet50:

```text
optimizer: SGD / Adam
lr:        1e-5 ... 1e-2
batch:     16 / 32 / 64
weight_decay: 1e-6 ... 1e-3
```

Для ViT-B/16 диапазон learning rate адаптирован под специфику transformer-модели.

> Важно: текущая версия ноутбука содержит конфигурационные search spaces для tuning, но основной показанный benchmark запускает заранее заданные конфигурации. Поэтому Optuna не следует считать обязательной runtime-зависимостью проекта без дополнительной интеграции optimization loop.

---

# 💾 Сохранение результатов

Лучший checkpoint сохраняется при улучшении validation accuracy:

```python
torch.save(
    model.state_dict(),
    save_path
)
```

История обучения также сохраняется:

```python
torch.save(
    history,
    history_path
)
```

Структура результатов:

```text
results/
├── resnet50_best.pth
├── resnet50_history.pt
├── efficientnet_b0_best.pth
├── efficientnet_b0_history.pt
├── mobilenet_v3_large_best.pth
├── mobilenet_v3_large_history.pt
├── vit_b_16_best.pth
└── vit_b_16_history.pt
```

---

# 🖥 Установка на Linux Mint

## 1. Системные зависимости

Установите Python, `venv`, Git, `wget` и `unzip`:

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git wget unzip
```

Проверьте Python:

```bash
python3 --version
```

Рекомендуется использовать современную версию Python 3.

---

## 2. Клонирование проекта

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_DIRECTORY>
```

Если проект уже скачан, просто перейдите в его директорию:

```bash
cd <PROJECT_DIRECTORY>
```

---

## 3. Создание virtual environment

```bash
python3 -m venv .venv
```

Активируйте окружение:

```bash
source .venv/bin/activate
```

После активации в терминале должен появиться префикс:

```text
(.venv)
```

---

## 4. Обновление pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

---

## 5. Установка Python-зависимостей

Основной набор зависимостей:

```bash
pip install numpy pandas matplotlib scikit-learn tqdm jupyter notebook
```

PyTorch и Torchvision:

```bash
pip install torch torchvision
```

После установки проверьте:

```bash
python -c "import torch, torchvision; print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__)"
```

Проверка CUDA:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Если `CUDA available: False`, код всё равно может работать на CPU, однако обучение больших моделей, особенно `ViT-B/16`, будет значительно медленнее.

Для GPU рекомендуется установить PyTorch с CUDA-сборкой, соответствующей вашей системе, используя официальный selector PyTorch:

https://pytorch.org/get-started/locally/

---

# 📦 Данные

Разные части лабораторной используют разные датасеты.

## Fashion-MNIST

В `Part 2` датасет загружается автоматически через `torchvision`:

```python
torchvision.datasets.FashionMNIST(
    root="fmnist",
    download=True
)
```

Отдельно скачивать Fashion-MNIST не требуется.

---

## Poetry

Для Shakespeare используется:

```text
sonnets.txt
```

Если файла нет, ноутбук скачивает его автоматически.

Для «Евгения Онегина» используется источник:

```text
https://raw.githubusercontent.com/attatrol/data_sources/master/onegin.txt
```

---

## Dogs Dataset

Для Part 5 используется датасет с 50 классами пород собак.

После распаковки ожидается структура:

```text
data/
├── train/
│   ├── breed_1/
│   ├── breed_2/
│   ├── ...
│   └── breed_50/
│
└── test/
    └── ...
```

В ноутбуке dataset загружается и распаковывается автоматически.

---

# ▶️ Запуск

Запустите Jupyter:

```bash
jupyter notebook
```

или:

```bash
jupyter lab
```

После запуска откройте нужный notebook.

### Part 2

```text
Lab2_DL_part2_overfitting.ipynb
```

### Part 3

```text
Lab2_DL_part3_poetry.ipynb
```

### Part 4 / Part 5

```text
Lab2_DL_parts_4_and_5_optional.ipynb
Lab2_DL_parts_5.ipynb
```

---

# 🚀 Быстрый запуск Part 5

После установки зависимостей:

```bash
source .venv/bin/activate
jupyter notebook Lab2_DL_parts_5.ipynb
```

В ноутбуке основная конфигурация находится в `EXPERIMENT`:

```python
EXPERIMENT = {
    "mode": "benchmark",
    "device": "cuda",
    "save_model": True,
    "save_history": True,
    "results_dir": "results"
}
```

Список моделей:

```python
list_of_models = [
    "resnet50",
    "efficientnet_b0",
    "mobilenet_v3_large",
    "vit_b_16"
]
```

После настройки можно запустить:

```python
all_results = train_all_models()
```

а затем:

```python
results_df = compare_models(all_results)
results_df
```

---

# 🔧 `.env`

Проект не использует обязательные секреты или переменные окружения.

Поэтому `.env` **не требуется**.

Основные параметры задаются непосредственно в notebook-конфигурации:

```python
EXPERIMENT = {
    "mode": "benchmark",
    "device": "cuda",
    "save_model": True,
    "save_history": True,
    "results_dir": "results"
}
```

Пути к датасетам также задаются в Python:

```python
data_root = "./data"
```

Рекомендуется заменить локальные абсолютные пути вида:

```text
/home/<user>/...
```

на относительные пути проекта.

---

# 📁 Рекомендуемая структура репозитория

После очистки Colab-метаданных и результатов notebook репозиторий можно организовать следующим образом:

```text
lab02_deep_learning/
│
├── README.md
│
├── notebooks/
│   ├── Lab2_DL_part2_overfitting.ipynb
│   ├── Lab2_DL_part3_poetry.ipynb
│   ├── Lab2_DL_parts_4_and_5_optional.ipynb
│   └── Lab2_DL_parts_5.ipynb
│
├── data/
│   ├── fmnist/
│   ├── train/
│   └── test/
│
├── results/
│   ├── *.pth
│   └── *_history.pt
│
├── models/
│   └── ...
│
└── .gitignore
```

> Большие датасеты и веса моделей не рекомендуется хранить непосредственно в Git-репозитории. Для них лучше использовать Git LFS, Releases или внешнее хранилище.

---

# 🧪 Тестирование и проверка

Это исследовательский проект на базе Jupyter notebooks, поэтому отдельного набора `pytest` unit tests в предоставленных материалах нет.

Вместо этого используются встроенные проверки и экспериментальная валидация.

Например, в Part 3 присутствуют assertions:

```python
assert len(text) == 100225
assert not any(
    [x in set(text) for x in string.ascii_uppercase]
)
```

Для Part 2 корректность модели проверяется через:

```text
Train Loss
Validation Loss
Validation Accuracy
```

Для Part 5 используется:

```text
Train Loss
Train Accuracy
Validation Loss
Validation Accuracy
Best Validation Accuracy
```

---

# 📈 Анализ результатов

Одно из основных наблюдений проекта — pretrained архитектуры значительно различаются по эффективности на одной и той же задаче.

В проведённом benchmark:

```text
ViT-B/16       ≈ 98.68%
ResNet50       ≈ 96.23%
EfficientNet   ≈ 91.77%
MobileNetV3    ≈ 90.79%
```

При этом сравнение проводится в едином pipeline, использующем одинаковый формат входных изображений, ImageNet normalization и сопоставимую двухфазную схему обучения.

Это делает Part 5 наиболее показательным экспериментом лабораторной: можно сравнить не только итоговую accuracy, но и динамику обучения, train/validation gap и поведение разных архитектур.

---

# 💡 Ключевой инженерный результат

Наиболее интересной частью проекта является не отдельная модель, а **унифицированный training pipeline для разных pretrained архитектур**.

Вместо написания отдельного training loop для каждой модели используются конфигурации:

```python
MODEL_CONFIGS
COMMON_BENCHMARK
COMMON_PHASE1
COMMON_PHASE2
EXPERIMENT
```

и универсальные функции:

```python
get_model()
get_train_config()
build_optimizer()
build_scheduler()
build_dataloaders()
train_epoch()
validate()
train_phase()
train_model()
train_all_models()
compare_models()
```

Таким образом, новая модель может быть подключена через конфигурацию, не переписывая весь pipeline обучения.

---

# 🧠 Pretrained Models

Готовые веса моделей и дополнительные артефакты экспериментов доступны во внешнем хранилище:

**Google Drive**

https://drive.google.com/drive/folders/1WEOIliZc_7mNfAYZFKXES4SE5QKPPmsI?usp=drive_link

В хранилище находятся веса обученных моделей, а также отдельная модель LSTM, обученная на тексте «Евгения Онегина».

> Большие model checkpoints намеренно не включаются непосредственно в README/Git-репозиторий.

---

# ⚠️ Примечания по запуску

## CUDA

Для больших моделей рекомендуется NVIDIA GPU.

Особенно это касается:

```text
ViT-B/16
ResNet50
EfficientNet-B0
```

Если CUDA недоступна, PyTorch автоматически может использовать CPU:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

---

## DataLoader

В notebook используется:

```python
num_workers=4
```

На системах с небольшим количеством CPU ядер это может привести к предупреждениям или снижению производительности.

Если запуск на Linux Mint работает нестабильно, попробуйте:

```python
num_workers=2
```

или:

```python
num_workers=0
```

---

## RAM / VRAM

`ViT-B/16` требует существенно больше памяти, чем лёгкие модели вроде MobileNetV3.

При нехватке VRAM уменьшите:

```python
batch_size
```

Например:

```python
batch_size = 8
```

---

# 🤝 Contributing

Проект является исследовательской лабораторной работой, но улучшения приветствуются.

Перед внесением изменений:

1. Сделайте Fork репозитория.
2. Создайте отдельную ветку.
3. Внесите изменения.
4. Проверьте notebook.
5. Создайте Pull Request.

Пример:

```bash
git checkout -b feature/new-experiment
```

После изменений:

```bash
git add .
git commit -m "Add new deep learning experiment"
git push origin feature/new-experiment
```

После этого создайте Pull Request.

### При добавлении нового эксперимента желательно указать:

- используемую модель;
- датасет;
- preprocessing;
- optimizer;
- learning rate;
- batch size;
- количество epochs;
- validation accuracy;
- краткий вывод.

---

# 📜 License

Лицензия для проекта отдельно не заявлена.

---

# 👨‍💻 Автор

**Lab 02 — Deep Learning**

Проект выполнен в формате исследовательской лабораторной работы и объединяет эксперименты с различными архитектурами нейронных сетей: от fully-connected и recurrent networks до современных pretrained Computer Vision моделей.

---

## ⭐ Итог

Проект демонстрирует полный путь от базовых экспериментов с нейронными сетями до практического использования современных pretrained моделей:

```text
Fully Connected
      │
      ▼
Overfitting
      │
      ▼
Regularization
      │
      ▼
Vanilla RNN
      │
      ▼
Text Generation
      │
      ▼
CNN / Computer Vision
      │
      ▼
Transfer Learning
      │
      ▼
Fine-Tuning
      │
      ▼
Model Benchmark
      │
      ├── ResNet50
      ├── EfficientNet-B0
      ├── MobileNetV3-Large
      └── ViT-B/16
```

Наиболее сильный результат экспериментов — `ViT-B/16` с **98.68% best validation accuracy**, за которым следует `ResNet50` с **96.23%**. При этом основная инженерная ценность Part 5 заключается в создании единого конфигурируемого pipeline, позволяющего обучать и сравнивать несколько архитектур без переписывания основной логики обучения.
