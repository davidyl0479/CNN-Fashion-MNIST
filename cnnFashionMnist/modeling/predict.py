from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import typer

from cnnFashionMnist.config import (
    DATASET_CONFIG,
    DEVICE_CONFIG,
    FIGURES_DIR,
    FILE_PATTERNS,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from cnnFashionMnist.dataset import create_data_loaders, prepare_datasets
from cnnFashionMnist.modeling.model import create_model

app = typer.Typer()

### File Purpose
### predict.py handles:
###
### 1. Loading trained models from checkpoints
### 2. Making predictions on test data
### 3. Evaluating model performance with comprehensive metrics
### 4. Saving predictions for analysis
### 5. Generating quick visualizations (confusion matrix)
### 6. Single image prediction (future feature)


def load_model(
    model_path: Path, model_type: str = "standard", device: Optional[torch.device] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load trained model from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model architecture
    model = create_model(model_type=model_type, num_classes=10)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"Model loaded from {model_path}")
    logger.info(f"Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
    logger.info(f"Best validation accuracy: {checkpoint.get('val_acc', 'unknown'):.2f}%")

    return model, checkpoint


def predict_batch(
    model: nn.Module, data_loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make predictions on a batch of data."""
    model.eval()  # Disable dropout, batch norm uses running stats
    all_predictions = []
    all_probabilities = []
    all_targets = []

    with torch.no_grad():  # # Disable gradient computation for speed/memory
        progress_bar = tqdm(data_loader, desc="Predicting")
        for data, targets in progress_bar:
            data, targets = data.to(device), targets.to(device)

            outputs = model(data)  # # Raw logits from model
            probabilities = F.softmax(outputs, dim=1)  # Convert to probabilities
            _, predictions = torch.max(outputs, 1)  # Get predicted class indices

            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return np.array(all_predictions), np.array(all_probabilities), np.array(all_targets)


def evaluate_model(
    predictions: np.ndarray, targets: np.ndarray, class_names: List[str]
) -> Dict[str, Any]:
    """Evaluate model predictions."""

    # Calculate accuracy
    accuracy = accuracy_score(targets, predictions)

    # Generate classification report
    report = classification_report(
        targets, predictions, target_names=class_names, output_dict=True
    )

    # Generate confusion matrix
    conf_matrix = confusion_matrix(targets, predictions)

    # Per-class accuracy
    per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

    evaluation_results = {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": conf_matrix,
        "per_class_accuracy": dict(zip(class_names, per_class_acc)),
        "class_names": class_names,
    }

    return evaluation_results


def save_predictions(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    targets: np.ndarray,
    class_names: List[str],
    output_path: Path,
):
    """Save predictions to CSV file."""

    # Create DataFrame with predictions
    results_df = pd.DataFrame(
        {
            "true_label": targets,  # 0, 1, 2, ...
            "predicted_label": predictions,  # 0, 1, 2, ...
            "true_class": [class_names[i] for i in targets],  # "T-shirt/top", ...
            "predicted_class": [class_names[i] for i in predictions],  # "Trouser", ...
            "correct": targets == predictions,  # True/False
        }
    )

    # Add probability columns
    for i, class_name in enumerate(class_names):
        results_df[f"prob_{class_name.replace('/', '_').replace('-', '_')}"] = probabilities[:, i]

    # Add confidence (max probability)
    results_df["confidence"] = np.max(probabilities, axis=1)

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    logger.info(f"Predictions saved to {output_path}")
    return results_df


def print_evaluation_summary(evaluation_results: Dict[str, Any]):
    """Print evaluation summary."""

    accuracy = evaluation_results["accuracy"]
    report = evaluation_results["classification_report"]
    per_class_acc = evaluation_results["per_class_accuracy"]

    logger.info(f"\n{'=' * 50}")
    logger.info("MODEL EVALUATION SUMMARY")
    logger.info(f"{'=' * 50}")
    logger.info(f"Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    logger.info("\nPer-class Accuracy:")

    for class_name, acc in per_class_acc.items():
        logger.info(f"  {class_name:<15}: {acc:.4f} ({acc * 100:.2f}%)")

    logger.info("\nMacro Average:")
    logger.info(f"  Precision: {report['macro avg']['precision']:.4f}")
    logger.info(f"  Recall:    {report['macro avg']['recall']:.4f}")
    logger.info(f"  F1-Score:  {report['macro avg']['f1-score']:.4f}")

    logger.info("\nWeighted Average:")
    logger.info(f"  Precision: {report['weighted avg']['precision']:.4f}")
    logger.info(f"  Recall:    {report['weighted avg']['recall']:.4f}")
    logger.info(f"  F1-Score:  {report['weighted avg']['f1-score']:.4f}")


def predict_single_image(
    model: nn.Module,
    image: torch.Tensor,
    class_names: List[str],
    device: torch.device,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Make prediction on a single image."""
    model.eval()

    with torch.no_grad():
        if len(image.shape) == 3:
            image = image.unsqueeze(0)  # Add batch dimension

        image = image.to(device)
        output = model(image)
        probabilities = F.softmax(output, dim=1)

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)

        predictions = {
            "top_predictions": [
                {
                    "class": class_names[int(idx.item())],
                    "probability": prob.item(),
                    "class_index": int(idx.item()),
                }
                for prob, idx in zip(top_probs[0], top_indices[0])
            ],
            "predicted_class": class_names[int(top_indices[0][0].item())],
            "confidence": top_probs[0][0].item(),
        }

    return predictions


def save_quick_confusion_matrix(
    evaluation_results: Dict[str, Any], model_type: str, save_plot: bool = True
) -> Optional[Path]:
    """Save a quick confusion matrix plot during prediction."""
    if not save_plot:
        return None

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    conf_matrix = evaluation_results["confusion_matrix"]
    class_names = evaluation_results["class_names"]

    # Normalize confusion matrix
    conf_matrix_norm = conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.title(f"Confusion Matrix - {model_type.title()} Model")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Use FILE_PATTERNS for consistent naming
    plot_filename = FILE_PATTERNS["plots"]["confusion_matrix"].format(model_type=model_type)
    plot_path = FIGURES_DIR / plot_filename

    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Confusion matrix plot saved to {plot_path}")
    return plot_path


################################          Workflow Visualization              ######################################
###############################     Load Trained Model
###############################             ↓
###############################     Prepare Test Data
###############################             ↓
###############################     ┌─────────────────────────────────┐
###############################     │         For Each Batch          │
###############################     ├─────────────────────────────────┤
###############################     │ 1. Forward pass (no gradients) │
###############################     │ 2. Apply softmax               │
###############################     │ 3. Get predictions             │
###############################     │ 4. Store results               │
###############################     └─────────────────────────────────┘
###############################             ↓
###############################     Calculate Metrics
###############################     ├─ Overall accuracy
###############################     ├─ Per-class metrics
###############################     ├─ Confusion matrix
###############################     └─ Classification report
###############################             ↓
###############################     Save Results
###############################     ├─ Predictions CSV
###############################     ├─ Evaluation pickle
###############################     └─ Confusion matrix plot
###############################             ↓
###############################     Display Summary
#######################################################################################################################


@app.command()
def main(
    model_path: Path = MODELS_DIR / FILE_PATTERNS["model"].format(model_type="standard"),
    model_type: str = "standard",
    batch_size: int = DEVICE_CONFIG.get("batch_size", 64),
    val_split: float = 0.1,
    random_seed: int = 42,
    num_workers: int = DEVICE_CONFIG["num_workers"],
    predictions_path: Path = PROCESSED_DATA_DIR
    / FILE_PATTERNS["predictions"].format(model_type="standard"),
    save_evaluation: bool = True,
    save_plots: bool = True,
):
    """Make predictions using trained Fashion-MNIST model."""

    logger.info("Starting Fashion-MNIST model inference...")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load class names from config or dataset info
    dataset_info_path = PROCESSED_DATA_DIR / "dataset_info.pkl"
    if dataset_info_path.exists():
        with open(dataset_info_path, "rb") as f:
            dataset_info = pickle.load(f)
        class_names = dataset_info["class_names"]
    else:
        class_names = DATASET_CONFIG["class_names"]

    # Prepare datasets if not already done
    if not dataset_info_path.exists():
        logger.info("Preparing datasets...")
        train_dataset, val_dataset, test_dataset = prepare_datasets(
            RAW_DATA_DIR, PROCESSED_DATA_DIR, val_split, random_seed
        )
    else:
        logger.info("Loading existing datasets...")
        train_dataset, val_dataset, test_dataset = prepare_datasets(
            RAW_DATA_DIR, PROCESSED_DATA_DIR, val_split, random_seed
        )

    # Create test data loader
    _, _, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size, num_workers
    )

    # Load model
    model, checkpoint = load_model(model_path, model_type, device)

    # Make predictions
    logger.info("Making predictions on test set...")
    predictions, probabilities, targets = predict_batch(model, test_loader, device)

    # Evaluate model
    evaluation_results = evaluate_model(predictions, targets, class_names)

    # Print evaluation summary
    print_evaluation_summary(evaluation_results)

    # Save predictions
    results_df = save_predictions(
        predictions, probabilities, targets, class_names, predictions_path
    )

    # Log prediction summary
    correct_count = results_df["correct"].sum()
    total_count = len(results_df)
    logger.info(
        f"Predictions saved: {correct_count}/{total_count} correct ({correct_count / total_count * 100:.2f}%)"
    )
    logger.info(f"Average confidence: {results_df['confidence'].mean():.3f}")

    # Save evaluation results
    if save_evaluation:
        eval_path = predictions_path.parent / FILE_PATTERNS["evaluation_results"].format(
            model_type=model_type
        )
        with open(eval_path, "wb") as f:
            pickle.dump(evaluation_results, f)
        logger.info(f"Evaluation results saved to {eval_path}")

    # Save confusion matrix plot
    if save_plots:
        plot_path = save_quick_confusion_matrix(evaluation_results, model_type, save_plots)
        if plot_path:
            logger.info(f"Confusion matrix plot saved to: {plot_path}")

    logger.success("Inference completed successfully!")
    logger.info(
        f"Test accuracy: {evaluation_results['accuracy']:.4f} ({evaluation_results['accuracy'] * 100:.2f}%)"
    )


@app.command()
def predict_image(
    image_path: Path,
    model_path: Path = MODELS_DIR / "fashion_mnist_standard.pth",
    model_type: str = "standard",
    top_k: int = 3,
):
    """Make prediction on a single image file."""

    # This function would need additional implementation for loading external images
    # For now, it's a placeholder for future enhancement
    logger.info("Single image prediction not implemented yet.")
    logger.info("Use the main prediction function for batch processing.")


if __name__ == "__main__":
    app()
