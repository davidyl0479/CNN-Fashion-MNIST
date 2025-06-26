from pathlib import Path
import pickle
import time
from typing import Any, Dict, Optional, Tuple

from loguru import logger
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import typer

from cnnFashionMnist.config import (
    DEVICE_CONFIG,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TRAINING_CONFIG,
)
from cnnFashionMnist.dataset import create_data_loaders, prepare_datasets
from cnnFashionMnist.modeling.losses import FocalLoss, WeightedFocalLoss
from cnnFashionMnist.modeling.model import create_model, initialize_weights

app = typer.Typer()

###
## File Purpose
# train.py handles:
# 1. Model training - The core training loop
# 2. Validation - Monitoring model performance
# 3. Early stopping - Preventing overfitting
# 4. Checkpointing - Saving model progress
# 5. Learning rate scheduling - Optimizing training dynamics
###


class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving."""

    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        # patience=7 : Wait 7 epochs without improvement before stopping
        # min_delta=0.001 : Minimum improvement to count as "better"
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="Training")
    for batch_idx, (data, targets) in enumerate(progress_bar):
        data, targets = data.to(device), targets.to(device)  # Move to GPU/CPU

        optimizer.zero_grad()  # Clear gradients from previous step
        outputs = model(data)  # Forward pass: predict
        loss = criterion(outputs, targets)  # Calculate loss
        loss.backward()  # Backward pass: compute gradients
        optimizer.step()  # Update weights

        # Track Metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)  # Get predicted class
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update progress bar
        progress_bar.set_postfix(
            {
                "Loss": f"{running_loss / (batch_idx + 1):.4f}",
                "Acc": f"{100.0 * correct / total:.2f}%",
            }
        )

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate_epoch(
    model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    """Validate model for one epoch."""
    model.eval()  # Disables dropout, batch norm uses running stats
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Saves memory, speeds up validation
        progress_bar = tqdm(val_loader, desc="Validation")
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)  # Only forward pass, no backward
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)  # Get predicted class
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar.set_postfix(
                {
                    "Loss": f"{running_loss / (batch_idx + 1):.4f}",
                    "Acc": f"{100.0 * correct / total:.2f}%",
                }
            )

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    train_acc: float,
    val_acc: float,
    model_path: Path,
    is_best: bool = False,
):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),  # Model weights
        "optimizer_state_dict": optimizer.state_dict(),  # Optimizer state
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "model_info": model.get_model_info() if hasattr(model, "get_model_info") else {},
    }

    # Save regular checkpoint
    torch.save(checkpoint, model_path)  # Regular checkpoint

    # Save best model separately
    if is_best:
        best_path = model_path.parent / f"best_{model_path.name}"
        torch.save(checkpoint, best_path)  # Best model so far
        logger.info(f"Best model saved to {best_path}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    early_stopping_patience: Optional[int] = None,
    save_every: Optional[int] = None,
    model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train the CNN model with smart defaults from config."""

    # Use config values as fallbacks
    epochs = epochs or TRAINING_CONFIG["epochs"]
    learning_rate = learning_rate or TRAINING_CONFIG["learning_rate"]
    early_stopping_patience = early_stopping_patience or TRAINING_CONFIG["early_stopping_patience"]
    save_every = save_every or TRAINING_CONFIG["save_every"]
    model_path = model_path or MODELS_DIR / "fashion_mnist_cnn.pth"

    # Setup device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Device: Use GPU if available, fallback to CPU
    model = model.to(device)
    logger.info(f"Training on device: {device}")

    # Initialize model weights
    initialize_weights(model)  # Weight initialization: Start with good random weights

    # Setup optimizer and loss function
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )  # Adam optimizer: Adaptive learning rate optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )  # Learning rate scheduler: Reduce LR when validation plateaus

    class_weights = torch.tensor([1.2, 0.5, 0.8, 0.7, 1.4, 0.5, 1.3, 0.6, 0.4, 0.5])
    criterion = WeightedFocalLoss(alpha=class_weights.to(device), gamma=2.0)

    # Early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience)

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rates": [],
    }

    best_val_acc = 0.0
    start_time = time.time()

    # Initialize variables in case epochs=0
    epoch = 0
    train_loss = 0.0
    val_loss = 0.0
    train_acc = 0.0
    val_acc = 0.0

    logger.info(f"Starting training for {epochs} epochs...")
    logger.info(
        f"Model info: {model.get_model_info() if hasattr(model, 'get_model_info') else 'N/A'}"
    )

    ###### Main Training Loop
    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Record metrics
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["learning_rates"].append(current_lr)

        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        logger.info(f"Learning Rate: {current_lr:.6f}")

        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        if (epoch + 1) % save_every == 0 or is_best:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                train_loss,
                val_loss,
                train_acc,
                val_acc,
                model_path,
                is_best,
            )

        # Early stopping
        if early_stopping(val_loss):
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    training_time = time.time() - start_time
    logger.success(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Save final model and training history
    save_checkpoint(
        model, optimizer, epoch, train_loss, val_loss, train_acc, val_acc, model_path, False
    )

    # Save training history
    history_path = model_path.parent / f"training_history_{model_path.stem}.pkl"
    with open(history_path, "wb") as f:
        pickle.dump(history, f)

    return history


############################################################################################
#######             Training Flow Visualization          ##################################
############## Start Training
##############       ↓
############## ┌─────────────────────────────────────────┐
############## │             For Each Epoch              │
############## ├─────────────────────────────────────────┤
############## │ 1. Train on all training batches       │
############## │    ├─ Forward pass                     │
############## │    ├─ Calculate loss                   │
############## │    ├─ Backward pass                    │
############## │    └─ Update weights                   │
############## │                                         │
############## │ 2. Validate on validation data         │
############## │    ├─ Forward pass only               │
############## │    └─ Calculate metrics               │
############## │                                         │
############## │ 3. Update learning rate scheduler      │
############## │ 4. Log progress                        │
############## │ 5. Save checkpoint (if needed)         │
############## │ 6. Check early stopping               │
############## └─────────────────────────────────────────┘
##############       ↓
############## End Training


@app.command()
def main(
    model_type: str = "standard",
    epochs: int = TRAINING_CONFIG["epochs"],
    batch_size: int = TRAINING_CONFIG["batch_size"],
    learning_rate: float = TRAINING_CONFIG["learning_rate"],
    val_split: float = TRAINING_CONFIG["val_split"],
    early_stopping_patience: int = TRAINING_CONFIG["early_stopping_patience"],
    random_seed: int = TRAINING_CONFIG["random_seed"],
    num_workers: int = DEVICE_CONFIG["num_workers"],
):
    """Train Fashion-MNIST CNN model."""

    logger.info("Starting Fashion-MNIST CNN training...")

    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Create directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare datasets
    logger.info("Preparing datasets...")
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        RAW_DATA_DIR, PROCESSED_DATA_DIR, val_split, random_seed
    )

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size, num_workers
    )

    # Create model
    logger.info(f"Creating {model_type} model...")
    model = create_model(model_type=model_type, num_classes=10)

    # Train model
    model_path = MODELS_DIR / f"fashion_mnist_{model_type}.pth"
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        model_path=model_path,
        early_stopping_patience=early_stopping_patience,
    )

    logger.success("Training completed successfully!")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Best validation accuracy: {max(history['val_acc']):.2f}%")


if __name__ == "__main__":
    app()
