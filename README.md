# Software Defect Prediction - MLOps Project

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/poetry-managed-cyan.svg)](https://python-poetry.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.4+-orange.svg)](https://mlflow.org/)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

A comprehensive MLOps project for predicting software defects using machine learning. This project implements a complete ML lifecycle with industry-standard tools and practices.

## 📋 Project Overview

This project predicts software defects based on various code metrics using a neural network classifier. It demonstrates a full MLOps pipeline including data processing, model training, experiment tracking, model serving, and CI/CD.

**Business Problem**: Predict whether a software module contains defects based on code metrics (loc, cyclomatic complexity, etc.)

**Dataset**: Software defect prediction dataset from Kaggle (Playground Series S3E23)

## 🚀 Features

- **End-to-End MLOps Pipeline**: Data ingestion → Processing → Training → Serving → Monitoring
- **Experiment Tracking**: MLflow for comprehensive experiment management
- **Model Versioning**: DVC for data versioning and reproducibility
- **Automated Workflows**: Makefile commands for all project operations
- **Automated Workflows**: GitHub Actions CI/CD pipelines
- **Quality Assurance**: Pre-commit hooks, testing, and code formatting
- **Containerization**: Docker support (for Triton Inference Server deployment)
- **Model Serving**: Multiple serving options (FastAPI, ONNX, TensorRT, Triton)
- **Documentation**: Auto-generated API documentation with Sphinx

## 🏗️ Project Structure

```
software-defect-prediction/
├── .github/workflows/          # GitHub Actions CI/CD pipelines
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main configuration
│   ├── data_config.yaml       # Data processing config
│   ├── model_config.yaml      # Model architecture config
│   └── training_config.yaml   # Training parameters
├── data/                      # Data directory
│   ├── raw/                   # Raw data (gitignored)
│   └── processed/             # Processed data (gitignored)
├── models/                    # Trained models (gitignored)
├── notebooks/                 # Jupyter notebooks for EDA
├── scripts/                   # Utility scripts
│   ├── download_data.py      # Kaggle data download
│   ├── data_preprocessing.py # Data processing pipeline
│   ├── exploratory_analysis.py # EDA and visualization
│   └── run_mlflow.py        # Script for run MLflow
│   └── convert_to_onnx.py   # Script for convert to ONNX format
├── software_defect_predictor/  # Source code
│   ├── data/                 # Data handling modules
│   │   ├── dataset.py       # PyTorch Dataset
│   │   └── datamodule.py    # PyTorch Lightning DataModule
│   ├── models/              # Model definitions
│   │   ├── model.py        # Neural network model
│   │   └── metric.py       # Custom metrics
│   ├── training/           # Training logic
│   │   └── trainer.py     # Model trainer
│   ├── inference/         # Inference utilities
│   │   └── predictor.py   # Model predictor
│   ├── pipelines/         # Main pipelines
│   │   ├── train_pipeline.py    # Training pipeline
│   │   └── inference_pipeline.py # Inference pipeline
│   └── utils/             # Utility functions
│       └── logging.py     # Logging configuration
├── tests/                  # Unit tests
├── docker/                # Docker configurations
├── triton_models/                # Triton Inference Server configs
├── .pre-commit-config.yaml # Pre-commit hooks
├── .dvcignore            # DVC ignore patterns
├── .gitignore           # Git ignore patterns
├── Makefile             # Project commands
├── pyproject.toml       # Poetry dependencies
├── poetry.lock         # Locked dependencies
└── README.md           # This file
```

## 🛠️ Technology Stack

| Category                  | Tools                                   |
|---------------------------|-----------------------------------------|
| **Language**              | Python 3.9+                             |
| **ML Framework**          | PyTorch, PyTorch Lightning              |
| **Configuration**         | Hydra, OmegaConf                        |
| **Experiment Tracking**   | MLflow                                  |
| **Data Versioning**       | DVC                                     |
| **Dependency Management** | Poetry                                  |
| **Code Quality**          | Black, Flake8, isort, mypy              |
| **Testing**               | pytest                                  |
| **Documentation**         | Sphinx                                  |
| **CI/CD**                 | GitHub Actions                          |
| **Containerization**      | Docker, Docker Compose                  |
| **Model Serving**         | FastAPI, MLflow, ONNX, TensorRT, Triton |
| **Monitoring**            | MLflow Tracking                         |

## ⚙️ Installation

### Prerequisites

- Python 3.9 or higher
- [Poetry](https://python-poetry.org/docs/#installation)
- [Git](https://git-scm.com/)
- [Kaggle CLI](https://github.com/Kaggle/kaggle-api) (for data download)
- [DVC](https://dvc.org/doc/install) (optional, for data versioning)
- [Docker](https://docs.docker.com/get-docker/) (optional, for Triton)

### Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/E-Kater/Binary-Classification-with-Software-Defects-Dataset.git
cd software-defect-prediction
```

2. **Install dependencies:**
```bash
make install
```

3. **Set up pre-commit hooks:**
```bash
make setup
```

4. **Activate the virtual environment:**
```bash
poetry env activate
```

### Manual Installation

```bash
# Install Poetry if not installed
curl -sSL https://install.python-poetry.org | python3 -

# Clone and setup project
git clone https://github.com/E-Kater/Binary-Classification-with-Software-Defects-Dataset.git
cd software-defect-prediction

# Install dependencies
poetry install

# Initialize DVC (optional)
dvc init

# Install pre-commit hooks
pre-commit install
```

## 📊 Dataset

The project uses the Software Defect Prediction dataset from Kaggle Playground Series S3E23.

### Data Features

The dataset contains 21 code metrics:
- `loc`: Lines of code
- `v(g)`: Cyclomatic complexity
- `ev(g)`: Essential complexity
- `iv(g)`: Design complexity
- `n`: Halstead program length
- `v`: Halstead volume
- `l`: Halstead program level
- `d`: Halstead difficulty
- `i`: Halstead intelligence
- `e`: Halstead effort
- `b`: Halstead bugs
- `t`: Halstead time estimator
- `lOCode`: Lines of code
- `lOComment`: Lines of comments
- `lOBlank`: Blank lines
- `locCodeAndComment`: Code and comment lines
- `uniq_Op`: Unique operators
- `uniq_Opnd`: Unique operands
- `total_Op`: Total operators
- `total_Opnd`: Total operands
- `branchCount`: Number of branches

**Target variable**: `defects` (TRUE/FALSE)

### Downloading Data

```bash
# Option 1: Using Kaggle API (recommended)
make download

# Option 2: Manual download
# 1. Download from Kaggle: https://www.kaggle.com/competitions/playground-series-s3e23
# 2. Place the CSV file in data/raw/
```

## 🚀 Usage

#### Data Download Using Kaggle API
make download

# Or using DVC (if already versioned)
make dvc-pull

### Complete Pipeline

Run the entire MLOps pipeline:

```bash
make full-pipeline
```

This will:
1. Download data from Kaggle
2. Preprocess and split the data
3. Train the model with experiment tracking
4. Evaluate on test set
5. Save the best model
6. Convert model to ONNX format
7. Run Fast API app

### Individual Steps

#### 1. Data Processing
```bash
make preprocess
```

#### 2. Exploratory Data Analysis
```bash
make explore
```

#### 3. Train Model
```bash
make train
# or with custom parameters
python software_defect_predictor/pipelines/train_pipeline.py model.learning_rate=0.0005 model.hidden_sizes="[128,64]"
```

#### 4. Run Inference
```bash
make api
```

#### 5. Launch MLflow UI
```bash
make mlflow
# Open http://localhost:5000 in your browser
```

### Model Training Options

#### Basic Training
```bash
python software_defect_predictor/pipelines/train_pipeline.py
```

## 🔧 Configuration

The project uses Hydra for configuration management. Key configuration files:

- **`configs/config.yaml`**: Main configuration
- **`configs/data_config.yaml`**: Data processing settings
- **`configs/model_config.yaml`**: Model architecture
- **`configs/training_config.yaml`**: Training parameters

### Example Configuration Override

```bash
# Train with custom parameters
python software_defect_predictor/pipelines/train_pipeline.py \
    model.learning_rate=0.0005 \
    model.hidden_sizes="[128,64,32]" \
    model.dropout_rate=0.4 \
    training.max_epochs=100
```

## 📈 Model Architecture

The project implements a neural network classifier:

```
Input (21 features)
    ↓
Linear(21 → 128)
    ↓
ReLU Activation
    ↓
Dropout(0.3)
    ↓
Linear(128 → 64)
    ↓
ReLU Activation
    ↓
Dropout(0.3)
    ↓
Linear(64 → 32)
    ↓
ReLU Activation
    ↓
Dropout(0.3)
    ↓
Linear(32 → 2)
    ↓
Output (defect/no defect)
```

### Key Features

- **Class Weighting**: Handles imbalanced datasets
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rate
- **Gradient Clipping**: Improves training stability
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1
- **Multiple Model Formats**: PyTorch, ONNX, TensorRT support

## 🧪 Testing

Run tests to ensure code quality:

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_model.py -v
```

## 🎯 Model Serving

### Option 1: FastAPI REST API
```bash
make api
# API available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### Option 2: MLflow Serving
```bash
# Serve a specific model version
make mlflow-ui
# Available at http://localhost:8001

### Tracked Information

- Parameters: All Hydra configuration parameters
- Metrics: Training/validation loss, F1, accuracy per epoch
- Artifacts: Models, configs, plots, logs
- Tags: Experiment metadata
```

### Option 3: Triton Inference Server (Production)
```bash
# Convert model to Triton format
make triton-convert

# Start Triton server
make triton-start

# Stop Triton client
make triton-stop
```

### Inference Examples

```python
# Python client example
from software_defect_prediction.inference.predictor import DefectPredictor

predictor = DefectPredictor("models/best_model.ckpt")
result = predictor.predict({
    "loc": 22.0,
    "v(g)": 3.0,
    # ... other features
})
```

```bash
# REST API call
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "loc": 22.0,
       "v(g)": 3.0,
       "ev(g)": 1.0,
       "iv(g)": 2.0,
       "n": 60.0,
       "v": 278.63,
       "l": 0.06,
       "d": 19.56,
       "i": 14.25,
       "e": 5448.79,
       "b": 0.09,
       "t": 302.71,
       "lOCode": 17,
       "lOComment": 1,
       "lOBlank": 1,
       "locCodeAndComment": 0,
       "uniq_Op": 16.0,
       "uniq_Opnd": 9.0,
       "total_Op": 38.0,
       "total_Opnd": 22.0,
       "branchCount": 5.0
     }'
```

## 📊 Experiment Tracking with MLflow

MLflow is integrated for comprehensive experiment tracking:

```bash
# Start MLflow UI
make mlflow-ui

# View experiments at http://localhost:5000
```

### Tracked Information

- **Parameters**: All configuration parameters
- **Metrics**: Training and validation metrics per epoch
- **Artifacts**: Models, logs, configuration files
- **Tags**: Experiment metadata
- **Plots**: Training curves, confusion matrices

## 🔄 CI/CD Pipeline

GitHub Actions workflows automate:

1. **Code Quality Checks**: Linting, formatting, type checking
2. **Unit Tests**: Automated testing with coverage
3. **Integration Tests**: End-to-end pipeline testing
4. **Model Training**: Scheduled retraining
5. **Deployment**: Docker image building and pushing


## 📁 Data Versioning with DVC

Track datasets and models with DVC:

```bash
# Track data files
dvc add data/raw/train.csv

# Track processed data
dvc add data/processed/train.csv data/processed/test.csv

# Push to remote storage
dvc push

# Pull data
dvc pull
```

## 🧹 Code Quality

### Pre-commit Hooks

Automated checks run before each commit:

```bash
# Install hooks
make setup

# Run manually
make lint

# Auto-fix issues
black software_defect_predictor/ tests/ scripts/
isort software_defect_predictor/ tests/ scripts/
```

### Code Formatting Standards

- **Black**: Code formatting (88 char line length)
- **isort**: Import sorting
- **Flake8**: Code style checking
- **mypy**: Type checking
- **pre-commit**: Automated git hooks

## 📈 Performance Monitoring

### Training Monitoring

```bash
# Monitor training logs
make monitor

# View MLflow metrics in real-time
make mlflow
```

### Model Performance

Key metrics tracked:
- **Accuracy**: Overall prediction correctness
- **Precision**: Quality of positive predictions
- **Recall**: Coverage of actual positives
- **F1-Score**: Balance of precision and recall

## 🔍 Debugging

### Common Issues

1. **Missing Data**: Ensure data is in `data/raw/train.csv`
2. **CUDA Errors**: Set `accelerator: cpu` in config for CPU-only machines
3. **Import Errors**: Run `poetry install` to ensure all dependencies
4. **Memory Issues**: Reduce `batch_size` in config
5. **Missing DVC remote**: Configure in .dvc/config
6. **Kaggle API error**: Set up ~/.kaggle/kaggle.json
7. **CUDA issues**: Set accelerator: cpu in config

### Debug Commands

```bash
# Check installation
make check_inference

```

## 📚 Documentation

### Generated Documentation Using Sphinx

```bash
# Generate project documentation
make  docs-gen
```

### Code Documentation

- Docstrings follow Google style
- Type hints throughout codebase
- Comments clarified some possible misunderstandings
- README files in each module
