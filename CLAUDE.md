# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CNN project for Fashion-MNIST dataset classification using the Cookiecutter Data Science template structure. The project is currently in template form with placeholder implementations that need to be replaced with actual CNN code.

## Development Commands

### Environment Setup
```bash
# Create conda environment
make create_environment
# Activate: conda activate CNN-Fashion-MNIST

# Install dependencies  
make requirements
```

### Code Quality
```bash
# Lint code using ruff
make lint

# Format code using ruff
make format

# Run tests
make test
# Single test: python -m pytest tests/test_data.py::test_code_is_tested
```

### Data Pipeline
```bash
# Download and prepare Fashion-MNIST dataset
make data
# Executes: python cnnFashionMnist/dataset.py
```

### Development Help
```bash
# Show all available make commands
make help
make  # default target shows help
```

## Architecture Structure

### Core Package: `cnnFashionMnist/`

**Configuration Management:**
- `config.py`: Centralized path management using Path objects
- All data paths defined as constants (RAW_DATA_DIR, PROCESSED_DATA_DIR, etc.)
- Integrated with loguru for structured logging
- tqdm integration for progress bars

**Data Pipeline Components:**
- `dataset.py`: Fashion-MNIST data loading and preprocessing (needs implementation)
- `features.py`: Feature engineering pipeline (needs implementation)  
- `modeling/train.py`: Model training entry point (needs implementation)
- `modeling/predict.py`: Model inference (needs implementation)
- `plots.py`: Visualization utilities (needs implementation)

**CLI Interface:**
All modules use Typer for command-line interfaces with consistent patterns:
```python
@app.command()
def main(input_path: Path = DEFAULT_PATH, output_path: Path = DEFAULT_PATH):
```

### Data Organization

Following Cookiecutter Data Science structure:
- `data/raw/`: Original Fashion-MNIST dataset
- `data/interim/`: Intermediate transformed data
- `data/processed/`: Final datasets for modeling
- `data/external/`: Third-party data sources
- `models/`: Saved model files (.pkl, .pth, etc.)
- `reports/figures/`: Generated visualizations
- `notebooks/`: Jupyter notebooks for exploration

### Configuration Details

**Python Environment:**
- Python 3.11 (strict requirement: `~=3.11.0`)
- Package built with flit_core

**Code Style:**
- Ruff for linting and formatting
- Line length: 99 characters
- Import sorting enabled
- First-party package: `cnnFashionMnist`

**Dependencies:**
Core ML stack: numpy, pandas, scikit-learn, matplotlib
Development: pytest, ruff, loguru, tqdm, typer
Interactive: jupyterlab, ipython, notebook

## Implementation Status

**Current State:** Template project with placeholder implementations
**Needs Implementation:** 
- Fashion-MNIST data loading in `dataset.py`
- CNN model architecture and training in `modeling/train.py` 
- Feature extraction for Fashion-MNIST in `features.py`
- Prediction pipeline in `modeling/predict.py`
- Visualization functions in `plots.py`
- Actual tests in `tests/test_data.py` (currently has failing placeholder)

**Ready Components:**
- Project structure and path management
- CLI interfaces with Typer
- Code quality tools configuration
- Development workflow with Make commands
- Logging and progress tracking setup

## Development Workflow

1. **Data Preparation:** Run `make data` to download Fashion-MNIST
2. **Feature Engineering:** Implement and run feature extraction
3. **Model Development:** Build CNN architecture in modeling modules
4. **Training:** Execute training pipeline via CLI or notebooks
5. **Evaluation:** Generate predictions and visualizations
6. **Quality Assurance:** Run `make lint` and `make test` before committing

## Key Integration Points

- All modules import paths from `cnnFashionMnist.config`
- Consistent error handling with loguru logger
- Progress tracking with tqdm in all long-running operations
- CLI access through Typer applications in each module
- Path management uses pathlib.Path throughout