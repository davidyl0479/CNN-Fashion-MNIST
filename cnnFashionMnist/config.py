from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
import torch

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Fashion-MNIST Dataset Configuration
DATASET_CONFIG = {
    "name": "Fashion-MNIST",  # Dataset identifier for logging and tracking
    "num_classes": 10,  # Total number of classification categories
    "input_shape": (1, 28, 28),  # Image dimensions: (channels, height, width) - grayscale 28x28
    "class_names": [  # Human-readable labels for each class index (0-9)
        "T-shirt/top",  # Class 0
        "Trouser",  # Class 1
        "Pullover",  # Class 2
        "Dress",  # Class 3
        "Coat",  # Class 4
        "Sandal",  # Class 5
        "Shirt",  # Class 6
        "Sneaker",  # Class 7
        "Bag",  # Class 8
        "Ankle boot",  # Class 9
    ],
    "mean": (0.5,),  # Normalization mean for pixel values (converts 0-255 to -1,1 range)
    "std": (0.5,),  # Normalization standard deviation for pixel values
}

# Model Configuration
MODEL_CONFIG = {
    "standard": {
        "name": "FashionMNISTCNN",
        "dropout_rate": 0.5,
        "num_classes": DATASET_CONFIG["num_classes"],
        "description": "Deep CNN with 3 conv blocks and batch normalization",
    },
    "simple": {
        "name": "SimpleFashionMNISTCNN",
        "num_classes": DATASET_CONFIG["num_classes"],
        "description": "Lightweight CNN with 2 conv blocks",
    },
}

# Training Configuration
TRAINING_CONFIG = {
    "batch_size": 64,  # Number of samples processed together in one forward/backward pass
    "epochs": 20,  # Number of complete passes through the entire training dataset
    "learning_rate": 0.001,  # Step size for gradient descent optimizer (Adam default)
    "weight_decay": 1e-4,  # L2 regularization strength to prevent overfitting
    "early_stopping_patience": 7,  # Stop training if validation loss doesn't improve for N epochs
    "lr_scheduler_patience": 3,  # Reduce learning rate if validation loss plateaus for N epochs
    "lr_scheduler_factor": 0.5,  # Factor to multiply learning rate by when reducing (new_lr = lr * factor)
    "val_split": 0.1,  # Fraction of training data to use for validation (10%)
    "random_seed": 42,  # Seed for reproducible random number generation
    "save_every": 5,  # Save model checkpoint every N epochs for recovery/analysis
}

# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    "train": {  # Augmentations applied only during training to increase data diversity
        "random_rotation": 10,  # Randomly rotate images by up to ±10 degrees
        "random_horizontal_flip": 0.5,  # 10% probability of flipping image horizontally
        "normalize": True,  # Apply normalization using dataset mean/std values
    },
    "val_test": {  # Minimal preprocessing for validation and test sets
        "normalize": True,  # Only apply normalization, no random augmentations
    },
}

# Device Configuration
DEVICE_CONFIG = {
    "use_cuda": torch.cuda.is_available(),
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "num_workers": 2,  # For data loading
    "pin_memory": True,  # For faster GPU transfer
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    "metrics": [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    ],  # Classification metrics to compute
    "confusion_matrix": True,  # Generate confusion matrix for detailed error analysis
    "per_class_metrics": True,  # Calculate metrics separately for each of the 10 classes
    "save_predictions": True,  # Save model predictions to file for further analysis
    "top_k_accuracy": [1, 3],  # Calculate Top-1 and Top-3 accuracy (correct prediction in top K)
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 300,
    "save_format": "png",
    "color_palette": "husl",
    "plot_style": "default",
    "font_size": 12,
}

# Logging Configuration
LOGGING_CONFIG = {
    "log_level": "INFO",
    "log_format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
    "log_training_metrics": True,
    "log_model_summary": True,
}

# File naming patterns
FILE_PATTERNS = {
    "model": "fashion_mnist_{model_type}.pth",
    "best_model": "best_fashion_mnist_{model_type}.pth",
    "training_history": "training_history_fashion_mnist_{model_type}.pkl",
    "evaluation_results": "evaluation_results_{model_type}.pkl",
    "predictions": "test_predictions_{model_type}.csv",
    "plots": {
        "training_history": "training_history_{model_type}.png",
        "confusion_matrix": "confusion_matrix_{model_type}.png",
        "class_performance": "class_performance_{model_type}.png",
        "sample_predictions": "sample_predictions_{model_type}.png",
        "class_distribution": "class_distribution.png",
    },
}

# Experiment tracking
EXPERIMENT_CONFIG = {
    "track_experiments": False,  # Set to True to enable experiment tracking
    "experiment_name": "fashion_mnist_cnn",
    "tags": ["cnn", "fashion-mnist", "classification"],
    "log_artifacts": True,
    "log_model": True,
}

# Performance benchmarks (for reference)
PERFORMANCE_BENCHMARKS = {
    "baseline_accuracy": 0.10,  # Random guessing (10 classes)
    "good_accuracy": 0.85,  # Good performance threshold
    "excellent_accuracy": 0.90,  # Excellent performance threshold
    "target_accuracy": 0.88,  # Project target
}

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

# Log configuration on import
logger.info(f"Device: {DEVICE_CONFIG['device']}")
logger.info(f"CUDA available: {DEVICE_CONFIG['use_cuda']}")
logger.info(f"Target accuracy: {PERFORMANCE_BENCHMARKS['target_accuracy']:.1%}")
