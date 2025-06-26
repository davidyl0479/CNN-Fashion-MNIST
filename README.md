# Fashion-MNIST CNN Classification Project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A comprehensive CNN project for Fashion-MNIST dataset classification featuring multiple model architectures, advanced training techniques, and comprehensive evaluation framework.

## 🎯 Project Overview

This project implements state-of-the-art CNN architectures for Fashion-MNIST classification with focus on:

- **Multiple CNN Architectures**: Standard CNN, Attention-enhanced CNN
- **Comprehensive Evaluation**: Statistical comparison, Per-class analysis, Feature extraction, Visualization
- **Research-Grade Analysis**: McNemar's test, Bootstrap confidence intervals, Dimensionality reduction

### 📊 Key Results

| Model | Test Accuracy | Parameters | Key Features |
|-------|--------------|------------|--------------|
| **Standard CNN** | **92.83%** | 865K | BatchNorm + Dropout baseline |
| **Attention CNN** | 92.82% | 870K | Channel attention + Focal loss |

*Performance on 10,000 Fashion-MNIST test samples*

## 🚀 Quick Start

### Environment Setup
```bash
# Create conda environment
make create_environment
conda activate CNN-Fashion-MNIST

# Install dependencies
make requirements
```

### Train Models
```bash
# Download Fashion-MNIST dataset
make data

# Train standard CNN model
python -m cnnFashionMnist.modeling.train

# Train attention-enhanced model
python -m cnnFashionMnist.modeling.train --model-type attention
```

### Run Complete Analysis
```bash
# Launch comprehensive Jupyter analysis
jupyter notebook notebooks/fashion_mnist_demo.ipynb
```

## 📁 Project Structure

```
├── README.md                    <- This file
├── CLAUDE.md                    <- Development guidance for AI assistants
├── Makefile                     <- Development automation commands
├── pyproject.toml              <- Package configuration
├── requirements.txt            <- Python dependencies
│
├── data/                       <- Fashion-MNIST dataset storage
│   ├── raw/                   <- Original Fashion-MNIST files
│   ├── interim/               <- Preprocessed data
│   ├── processed/             <- Model-ready datasets
│   └── external/              <- Third-party data
│
├── models/                     <- Trained model artifacts
│   ├── fashion_mnist_demo.pth          <- Standard CNN model
│   ├── best_fashion_mnist_demo.pth     <- Best validation checkpoint
│   ├── fashion_mnist_attention.pth     <- Attention-enhanced model
│   └── fashion_mnist_*.pth             <- Other model variants
│
├── notebooks/                  <- Jupyter analysis notebooks
│   └── fashion_mnist_demo.ipynb       <- Complete project demonstration
│
├── reports/figures/            <- Generated visualizations
│   ├── training_history_*.png         <- Training curves
│   ├── confusion_matrix_*.png         <- Classification analysis
│   ├── class_performance_*.png        <- Per-class metrics
│   ├── model_comparison_*.png         <- Model evaluation plots
│   └── feature_analysis_*.png         <- Feature visualization
│
└── cnnFashionMnist/           <- Source code package
    ├── __init__.py
    ├── config.py              <- Configuration and paths
    ├── dataset.py             <- Data loading and preprocessing
    ├── features.py            <- Feature extraction and analysis
    ├── plots.py               <- Visualization utilities
    └── modeling/
        ├── __init__.py
        ├── model.py           <- CNN architecture definitions
        ├── train.py           <- Training pipeline
        └── predict.py         <- Model inference and evaluation
```

## 🏗️ Architecture Details

### Standard CNN Architecture
```python
FashionMNISTCNN(
  # Convolutional Feature Extraction
  (conv1-2): Conv2d(1→32) + BatchNorm + MaxPool + Dropout(0.25)
  (conv3-4): Conv2d(32→64) + BatchNorm + MaxPool + Dropout(0.25)  
  (conv5): Conv2d(64→128) + BatchNorm + MaxPool + Dropout(0.25)
  
  # Classification Head
  (fc1): Linear(1152→512) + BatchNorm + Dropout(0.5)
  (fc2): Linear(512→256) + BatchNorm + Dropout(0.5)
  (fc3): Linear(256→10)  # Output layer
)
```

**Key Features:**
- **Progressive channel expansion**: 1 → 32 → 64 → 128 channels
- **Spatial reduction**: 28×28 → 14×14 → 7×7 → 3×3 feature maps
- **Regularization**: BatchNorm + Dropout at every level
- **Parameter efficiency**: 865K parameters with 99% spatial compression

### Attention-Enhanced CNN
- **Channel Attention**: SE-blocks after conv3 and conv5
- **Focal Loss**: Addresses class imbalance with γ=2, α=0.25
- **Class Weighting**: Automatic balancing for challenging classes
- **Same base architecture** with attention overlays

## 📈 Training Configuration

```python
# Core Training Settings
TRAINING_CONFIG = {
    "epochs": 20,
    "batch_size": 64,
    "learning_rate": 0.001,
    "early_stopping_patience": 7,
    "val_split": 0.1,
    "random_seed": 42
}

# Data Augmentation
AUGMENTATION_CONFIG = {
    "random_rotation": 10,           # ±10° rotation
    "random_horizontal_flip": 0.5,   # 50% flip probability
    "normalization": (0.5, 0.5)     # [0,1] → [-1,1] scaling
}

# Optimizer & Scheduling
- Optimizer: Adam (β₁=0.9, β₂=0.999)
- LR Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
- Loss: CrossEntropy (standard) / Focal Loss (attention model)
```

## 📊 Comprehensive Evaluation

### Statistical Analysis Framework

The project includes rigorous statistical evaluation:

1. **McNemar's Test**: Paired model comparison with exact p-values
2. **Bootstrap Confidence Intervals**: 95% CI for accuracy differences  
3. **Per-Class Metrics**: Precision, Recall, F1-score for each fashion category
4. **Confusion Matrix Analysis**: Detailed misclassification patterns

### Key Evaluation Insights

**Class Performance Rankings:**
```
Highest Accuracy:  Trouser (98.7%), Sneaker (98.2%), Bag (98.4%)
Moderate Accuracy: Dress (93.5%), Pullover (90.3%), Ankle boot (95.6%)
Challenging:       Shirt (76.3%), Coat (85.7%), T-shirt/top (87.9%)
```

**Model Comparison Results:**
- **Standard vs Attention**: No statistically significant difference (p > 0.05)
- **Generalization**: All models show excellent validation→test consistency
- **Efficiency**: Standard CNN provides best accuracy-per-parameter ratio

## 🔬 Advanced Features

### Feature Analysis Pipeline
- **Multi-layer feature extraction**: conv1, conv3, conv5, fc1 activations
- **Dimensionality reduction**: PCA and t-SNE visualization of learned features
- **Clustering analysis**: K-means clustering with 86.3% purity on fc1 features
- **Feature statistics**: Progression analysis showing perfect CNN hierarchy

### Visualization Suite
- **Training History**: Loss/accuracy curves with early stopping analysis
- **Confusion Matrices**: Normalized and raw count visualizations
- **Class Performance**: Multi-metric comparison charts
- **Feature Spaces**: 2D projections of high-dimensional CNN features
- **Model Comparisons**: Side-by-side statistical evaluation

## 🛠️ Development Workflow

### Code Quality
```bash
# Lint code using ruff
make lint

# Format code using ruff  
make format

# Run tests
make test
```

### Available Commands
```bash
# Environment and dependencies
make create_environment    # Create conda environment
make requirements         # Install Python packages

# Data pipeline
make data                # Download Fashion-MNIST dataset

# Development
make lint                # Check code style
make format              # Auto-format code  
make test                # Run test suite
make help                # Show all available commands
```

## 📋 Requirements

### Core Dependencies
- **Python**: 3.11+
- **PyTorch**: 2.0+ (with torchvision)
- **NumPy**: Scientific computing
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **Scikit-learn**: ML utilities
- **Jupyter**: Interactive analysis

### Development Tools
- **Ruff**: Code formatting and linting
- **Pytest**: Testing framework
- **Loguru**: Structured logging
- **TQDM**: Progress bars
- **Typer**: CLI interfaces

### Optional Dependencies
- **CUDA**: GPU acceleration (recommended)
- **Statsmodels**: Advanced statistical tests

## 🎓 Educational Value

This project serves as a comprehensive example of:

1. **Modern CNN Architecture Design**: From basic convolutions to attention mechanisms
2. **Production ML Pipeline**: Data loading → Training → Evaluation → Deployment
3. **Rigorous Evaluation**: Statistical testing and significance analysis
4. **Research Methodology**: Systematic ablation studies and comparison
5. **Code Quality**: Type hints, documentation, testing, and CI/CD ready

### Key Learning Outcomes
- **CNN Design Principles**: Feature hierarchy, regularization, architectural choices
- **Training Best Practices**: Data augmentation, early stopping, learning rate scheduling  
- **Evaluation Methodology**: Statistical significance, confidence intervals, error analysis
- **Feature Understanding**: Visualization and interpretation of learned representations
- **Project Organization**: Cookiecutter Data Science structure and workflows

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)  
5. Open a Pull Request

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub.

---

**Built with ❤️ using PyTorch and the Cookiecutter Data Science template**