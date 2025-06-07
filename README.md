# Oxford Flowers Classification Project

## Project Overview

This project focuses on classifying plants based on their photographs. It uses the Oxford Flowers 102 dataset, which contains 102 flower categories with diverse images. The main goal is to build a model that accurately classifies plant images into their respective classes.

### Problem Statement

- **Input and output data:**
  - The dataset feature structure is:
    ```
    FeaturesDict({
      'file_name': Text(shape=(), dtype=string),
      'image': Image(shape=(None,None,3), dtype=uint8),
      'label': ClassLabel(shape=(), dtype=int64, num_classes=102)
    })
    ```
  - Input data: `Image(shape=(None,None,3), dtype=uint8)`
  - Output data: `ClassLabel(shape=(), dtype=int64, num_classes=102)`
- **Metrics:**

  - Loss: a measure of how well the model's predictions match the true labels during training or validation. Lower loss indicates better fit.
  - Accuracy: the proportion of correctly classified samples among all samples.
  - F1-score: the harmonic mean of precision and recall, balancing the trade-off between FP and FN.

  As expected, Accuracy and F1-score are close to 90%

- **Validation**
  In the original dataset, the train/validation split is already done. For reproducability, the value of random_state is fixed.

- **Data:**
  The dataset consists of 102 flower categories, wide-spread in the UK. Every class contains 40-258 images. Training set, validation set : 10 images for each set, test set: 6149 images (at least 20 images for each class).

  - **Specifics:**
    - Images within one class may differ from each other very much
    - Images from different classes may be very similar to each other
    - A lot of classes (102)
  - **Challenges:**
    - Small training set (10 images per class for training and validation).
    - High variability within classes.
    - No additional metadata available.

  Link: [https://huggingface.co/datasets/nkirschi/oxford-flowers]

### Model

ResNet-50 is a convolutional neural network that uses residual blocks to address the vanishing gradient problem. It consists of 50 layers and is designed for image classification

Link: [https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html]

---

## Setup

### Requirements

- Python 3.12
- Poetry for dependency management

### Installation

0. Install pyenv if missing, install Python 3.12, install poetry if missing:

```
command -v pyenv >/dev/null 2>&1 || (curl https://pyenv.run | bash && export PATH="$HOME/.pyenv/bin:$PATH" && eval "$(pyenv init --path)" && eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)"); \
export PATH="$HOME/.pyenv/bin:$PATH"; eval "$(pyenv init --path)"; eval "$(pyenv init -)"; eval "$(pyenv virtualenv-init -)"; \
pyenv install -s 3.12.3; pyenv local 3.12.3; \
command -v poetry >/dev/null 2>&1 || (curl -sSL https://install.python-poetry.org | python3 - && export PATH="$HOME/.local/bin:$PATH"); \
export PATH="$HOME/.local/bin:$PATH"; \
```

1. Clone the repository:

```
git clone https://github.com/cotucotudu/oxford-flowers-classification
cd oxford-flowers-classification
```

2. Install dependencies using Poetry:

```
poetry install
```

3. Activate Poetry virtual environment (optional):

```
poetry env activate
```

---

## Data Preparation

The `data.py` script downloads the Oxford Flowers 102 dataset, organizes images into folders, and prepares train, validation, and test splits.

To run data preparation:

```
poetry run python plants_classification/data.py data.batch_size=64 data.resize=300
```

### Adjustable Parameters (Hydra):

```
data:
  raw_dir: data/raw
  processed_dir: data/processed
  batch_size: 32
  num_workers: 4
  resize: 256
  crop_size: 224
```

---

## Train

0. If the mlflow server is not running yet, launch it:

```
poetry run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 8080
```

1. To start training the model, run:

```
poetry run python plants_classification/train.py data.batch_size=64 model.learning_rate=0.0005 training.max_epochs=30
```

### Adjustable parameters (Hydra):

```
data:
  data_dir: "../data"
  batch_size: 32
  num_workers: 4
  resize: 256
  crop_size: 224

model:
  num_classes: 102
  learning_rate: 0.001
  freeze_backbone: true

training:
  max_epochs: 50
  accelerator: "auto"
  devices: 1
  checkpoint_dir: "checkpoints"

logging:
  experiment_name: "flower_classification"
  mlflow_tracking_uri: "http://127.0.0.1:8080"
  plots_dir: "../plots"
```

---

## Inference

To run inference on new images, use:

```
poetry run python plants_classification/infer.py 'infer.image_path="../data/flowers-102/jpg/image_00001.jpg"'
```

- Input data format: path to a JPG image.
- Output: top-5 predicted classes with probabilities.

### Adjustable parameters (Hydra):

```
infer:
  model:
    checkpoint_path: "checkpoints/best-epoch=1.ckpt"
  image_path: "../data/flowers-102/jpg/image_00001.jpg"
  top_k: 5
  preprocess:
    resize: 256
    crop_size: 224
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
```

### Example Output

```
Top-5 preds for ../data/flowers-102/jpg/image_00001.jpg:
Class 76: probability 0.0996
Class 70: probability 0.0439
Class 12: probability 0.0349
Class 40: probability 0.0259
Class 48: probability 0.0256
```

---

## Export to ONNX

To export your trained model to ONNX, use:

```
poetry run python plants_classification/export_onnx.py model.learning_rate=0.01 model.freeze_backbone=false
```

### Adjustable parameters (Hydra):

```
export:
  model:
    checkpoint_path: "checkpoints/best-epoch=1.ckpt"
    onnx_output: "model.onnx"
    img_size: 300
    learning_rate: 0.0005
    freeze_backbone: true
    num_classes: 102
```

---

## Contact and Support

If you have any questions or suggestions, please open an issue in the repository or contact me directly
