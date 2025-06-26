from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from tqdm import tqdm
import typer

from cnnFashionMnist.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from cnnFashionMnist.dataset import get_transforms, prepare_datasets

app = typer.Typer()

plt.style.use("default")
sns.set_palette("husl")

### File Purpose
### plots.py handles:
###
### 1. Dataset visualization - Understanding the Fashion-MNIST data
### 2. Training progress plots - Monitoring learning curves
### 3. Model evaluation plots - Performance analysis
### 4. Data augmentation examples - Visualizing preprocessing
### 5. Comprehensive reporting - Professional-quality figures


def plot_class_distribution(
    dataset_info: Dict[str, Any], output_path: Optional[Path] = None, show_inline: bool = True
):
    """Plot the distribution of classes in the dataset.

    Args:
        dataset_info: Dictionary with dataset information
        output_path: Optional path to save the plot. If None, plot is not saved.
        show_inline: Whether to display plot inline (for Jupyter notebooks)

    Returns:
        Dict with dataset statistics
    """

    class_names = dataset_info["class_names"]
    train_size = dataset_info["train_size"]
    val_size = dataset_info["val_size"]
    test_size = dataset_info["test_size"]

    # Since Fashion-MNIST has balanced classes, each class has equal samples
    samples_per_class = {
        "Training": train_size // 10,
        "Validation": val_size // 10,
        "Test": test_size // 10,
    }

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: Overall dataset split
    sizes = [train_size, val_size, test_size]
    labels = ["Training", "Validation", "Test"]
    colors = ["#ff9999", "#66b3ff", "#99ff99"]

    ax1.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
    ax1.set_title("Dataset Split Distribution")

    # Plot 2: Samples per split
    x = np.arange(len(labels))
    bars = ax2.bar(x, [samples_per_class[label] for label in labels], color=colors)
    ax2.set_xlabel("Dataset Split")
    ax2.set_ylabel("Samples per Class")
    ax2.set_title("Samples per Class in Each Split")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)

    # Plot 3: Class names and their equal distribution
    class_colors = plt.cm.get_cmap("tab10")(np.arange(len(class_names)))
    bars3 = ax3.bar(
        range(len(class_names)),
        [train_size // len(class_names)] * len(class_names),
        color=class_colors,
        alpha=0.7,
    )
    ax3.set_xlabel("Fashion-MNIST Classes")
    ax3.set_ylabel("Samples per Class (Training)")
    ax3.set_title("Class Distribution in Training Set")
    ax3.set_xticks(range(len(class_names)))
    ax3.set_xticklabels(class_names, rotation=45, ha="right")

    # Add value labels on bars (plot 2)
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0, height, f"{int(height)}", ha="center", va="bottom"
        )

    # Add value labels on bars (plot 3)
    for bar in bars3:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0, height, f"{int(height)}", ha="center", va="bottom"
        )

    plt.tight_layout()

    # Save to file if output_path provided
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Class distribution plot saved to {output_path}")

    # Show inline for Jupyter notebooks
    if show_inline:
        plt.show()
    else:
        plt.close()

    # Return useful statistics
    return {
        "total_samples": train_size + val_size + test_size,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "samples_per_class": samples_per_class,
        "num_classes": len(class_names),
    }


def plot_sample_images(
    dataset,
    class_names: List[str],
    output_path: Optional[Path] = None,
    num_samples: int = 20,
    show_inline: bool = True,
):
    """Plot sample images from the dataset.

    Args:
        dataset: PyTorch dataset to sample from
        class_names: List of class names
        output_path: Optional path to save the plot. If None, plot is not saved.
        num_samples: Number of samples to show (currently fixed at 2 per class)
        show_inline: Whether to display plot inline (for Jupyter notebooks)

    Returns:
        Dict with sampling statistics
    """

    # Get samples from each class
    class_samples = {i: [] for i in range(len(class_names))}

    # Collect samples with progress bar
    dataset_iter = enumerate(dataset)
    if hasattr(dataset, "__len__"):
        dataset_iter = tqdm(dataset_iter, total=min(1000, len(dataset)), desc="Collecting samples")

    for i, (image, label) in dataset_iter:
        if len(class_samples[label]) < 2 and i < 1000:  # Limit search to first 1000 samples
            class_samples[label].append((image, label))

        # Stop when we have enough samples
        if all(len(samples) >= 2 for samples in class_samples.values()):
            break

    # Create subplot grid (10 columns x 2 rows)
    fig, axes = plt.subplots(2, len(class_names), figsize=(2 * len(class_names), 6))
    if len(class_names) == 1:
        axes = axes.reshape(-1, 1)

    for class_idx, class_name in enumerate(class_names):
        for sample_idx in range(2):
            ax = axes[sample_idx, class_idx]

            if len(class_samples[class_idx]) > sample_idx:
                image, label = class_samples[class_idx][sample_idx]

                # Convert tensor to numpy and handle normalization
                if isinstance(image, torch.Tensor):
                    if len(image.shape) == 3:
                        image_np = image.squeeze().numpy()
                    else:
                        image_np = image.numpy()

                    # Denormalize if needed (assuming normalization with mean=0.5, std=0.5)
                    if image_np.min() < 0:
                        image_np = (image_np + 1) / 2  # Convert from [-1,1] to [0,1]
                else:
                    image_np = np.array(image)

                ax.imshow(image_np, cmap="gray")
                ax.set_title(f"{class_name}" if sample_idx == 0 else "")

            ax.axis("off")

    plt.suptitle("Sample Images from Fashion-MNIST Dataset", fontsize=16)
    plt.tight_layout()

    # Save to file if output_path provided
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Sample images plot saved to {output_path}")

    # Show inline for Jupyter notebooks
    if show_inline:
        plt.show()
    else:
        plt.close()

    # Return sampling statistics
    return {
        "classes_sampled": len([k for k, v in class_samples.items() if len(v) > 0]),
        "total_samples": sum(len(v) for v in class_samples.values()),
        "samples_per_class": {
            class_names[k]: len(v) for k, v in class_samples.items() if len(v) > 0
        },
    }


def plot_training_history(
    history: Dict[str, List], output_path: Optional[Path] = None, show_inline: bool = True
):
    """Plot training history curves.

    Args:
        history: Training history dictionary with loss/accuracy data
        output_path: Optional path to save the plot. If None, plot is not saved.
        show_inline: Whether to display plot inline (for Jupyter notebooks)

    Returns:
        Dict with best epoch and accuracy information
    """

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Loss curves
    ax1.plot(epochs, history["train_loss"], "b-", label="Training Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Validation Loss")
    ax1.set_title("Model Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Accuracy curves
    ax2.plot(epochs, history["train_acc"], "b-", label="Training Accuracy")
    ax2.plot(epochs, history["val_acc"], "r-", label="Validation Accuracy")
    ax2.set_title("Model Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    # Plot 3: Learning rate schedule
    ax3.plot(epochs, history["learning_rates"], "g-", label="Learning Rate")
    ax3.set_title("Learning Rate Schedule")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Learning Rate")
    ax3.set_yscale("log")
    ax3.legend()
    ax3.grid(True)

    # Plot 4: Validation accuracy with best point
    ax4.plot(epochs, history["val_acc"], "r-", label="Validation Accuracy")
    best_epoch = np.argmax(history["val_acc"]) + 1
    best_acc = max(history["val_acc"])
    ax4.plot(best_epoch, best_acc, "ro", markersize=10, label=f"Best: {best_acc:.2f}%")
    ax4.set_title("Validation Accuracy Progress")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Accuracy (%)")
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()

    # Save to file if output_path provided
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Training history plot saved to {output_path}")

    # Show inline for Jupyter notebooks
    if show_inline:
        plt.show()
    else:
        plt.close()

    # Return useful information
    return {
        "best_epoch": best_epoch,
        "best_accuracy": best_acc,
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
    }


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: List[str],
    output_path: Optional[Path] = None,
    normalize: bool = True,
    show_inline: bool = True,
):
    """Plot confusion matrix.

    Args:
        conf_matrix: Confusion matrix as numpy array
        class_names: List of class names for labels
        output_path: Optional path to save the plot. If None, plot is not saved.
        normalize: Whether to normalize the confusion matrix
        show_inline: Whether to display plot inline (for Jupyter notebooks)

    Returns:
        Dict with confusion matrix statistics
    """

    if normalize:
        conf_matrix_norm = conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
        title = "Normalized Confusion Matrix"
        fmt = ".2f"
    else:
        conf_matrix_norm = conf_matrix
        title = "Confusion Matrix"
        fmt = "d"

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        conf_matrix_norm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save to file if output_path provided
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Confusion matrix plot saved to {output_path}")

    # Show inline for Jupyter notebooks
    if show_inline:
        plt.show()
    else:
        plt.close()

    # Return useful statistics
    return {
        "total_samples": np.sum(conf_matrix),
        "accuracy": np.trace(conf_matrix) / np.sum(conf_matrix),
        "per_class_accuracy": np.diag(conf_matrix) / np.sum(conf_matrix, axis=1),
        "normalized_matrix": conf_matrix_norm if normalize else None,
        "raw_matrix": conf_matrix,
    }


def plot_model_predictions(
    predictions_df: pd.DataFrame,
    class_names: List[str],
    output_path: Optional[Path] = None,
    num_samples: int = 20,
    show_inline: bool = True,
):
    """Plot model predictions with confidence scores.

    Args:
        predictions_df: DataFrame with prediction results
        class_names: List of class names
        output_path: Optional path to save the plot. If None, plot is not saved.
        num_samples: Number of samples to display
        show_inline: Whether to display plot inline (for Jupyter notebooks)

    Returns:
        Dict with prediction statistics
    """

    # Get both correct and incorrect predictions
    correct_preds = predictions_df[predictions_df["correct"]].head(num_samples // 2)
    incorrect_preds = predictions_df[~predictions_df["correct"]].head(num_samples // 2)

    # Combine samples
    sample_preds = pd.concat([correct_preds, incorrect_preds])

    if len(sample_preds) == 0:
        logger.warning("No predictions found to plot")
        return {"error": "No predictions found"}

    # Create subplot grid
    n_cols = 5
    n_rows = (len(sample_preds) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (_, row) in enumerate(sample_preds.iterrows()):
        if idx >= num_samples:
            break

        ax = axes[idx // n_cols, idx % n_cols]

        # Create a simple placeholder image (since we don't have the actual images in the DataFrame)
        # In a real implementation, you would load the actual image data
        ax.text(0.5, 0.5, f"Image {idx}", ha="center", va="center", transform=ax.transAxes)

        # Set title with prediction info
        true_class = row["true_class"]
        pred_class = row["predicted_class"]
        confidence = row["confidence"]
        is_correct = row["correct"]

        color = "green" if is_correct else "red"
        title = f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}"

        ax.set_title(title, color=color, fontsize=10)
        ax.axis("off")

    # Hide empty subplots
    for idx in range(len(sample_preds), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    plt.suptitle("Model Predictions (Green=Correct, Red=Incorrect)", fontsize=16)
    plt.tight_layout()

    # Save to file if output_path provided
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Model predictions plot saved to {output_path}")

    # Show inline for Jupyter notebooks
    if show_inline:
        plt.show()
    else:
        plt.close()

    # Return prediction statistics
    return {
        "total_displayed": len(sample_preds),
        "correct_displayed": len(correct_preds),
        "incorrect_displayed": len(incorrect_preds),
        "average_confidence": sample_preds["confidence"].mean(),
        "confidence_correct": correct_preds["confidence"].mean() if len(correct_preds) > 0 else 0,
        "confidence_incorrect": incorrect_preds["confidence"].mean()
        if len(incorrect_preds) > 0
        else 0,
    }


def plot_class_performance(
    evaluation_results: Dict[str, Any],
    output_path: Optional[Path] = None,
    show_inline: bool = True,
):
    """Plot per-class performance metrics.

    Args:
        evaluation_results: Dictionary with evaluation results
        output_path: Optional path to save the plot. If None, plot is not saved.
        show_inline: Whether to display plot inline (for Jupyter notebooks)

    Returns:
        Dict with performance statistics
    """

    class_names = evaluation_results["class_names"]
    report = evaluation_results["classification_report"]

    # Extract metrics for each class
    metrics = {"precision": [], "recall": [], "f1-score": []}

    for class_name in class_names:
        class_report = report[class_name]
        metrics["precision"].append(class_report["precision"])
        metrics["recall"].append(class_report["recall"])
        metrics["f1-score"].append(class_report["f1-score"])

    # Create subplot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # Plot 1: Bar chart of metrics
    x = np.arange(len(class_names))
    width = 0.25

    ax1.bar(x - width, metrics["precision"], width, label="Precision", alpha=0.8)
    ax1.bar(x, metrics["recall"], width, label="Recall", alpha=0.8)
    ax1.bar(x + width, metrics["f1-score"], width, label="F1-Score", alpha=0.8)

    ax1.set_xlabel("Class")
    ax1.set_ylabel("Score")
    ax1.set_title("Per-Class Performance Metrics")
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Plot 2: Per-class accuracy
    per_class_acc = list(evaluation_results["per_class_accuracy"].values())
    bars = ax2.bar(class_names, per_class_acc, alpha=0.8, color="skyblue")
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Per-Class Accuracy")
    ax2.set_xticklabels(class_names, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # Add value labels on bars
    for bar, acc in zip(bars, per_class_acc):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    # Save to file if output_path provided
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Class performance plot saved to {output_path}")

    # Show inline for Jupyter notebooks
    if show_inline:
        plt.show()
    else:
        plt.close()

    # Return performance statistics
    return {
        "best_class": class_names[np.argmax(per_class_acc)],
        "worst_class": class_names[np.argmin(per_class_acc)],
        "avg_precision": np.mean(metrics["precision"]),
        "avg_recall": np.mean(metrics["recall"]),
        "avg_f1_score": np.mean(metrics["f1-score"]),
        "avg_accuracy": np.mean(per_class_acc),
        "class_metrics": dict(
            zip(class_names, zip(metrics["precision"], metrics["recall"], metrics["f1-score"]))
        ),
    }


def generate_confusion_matrix_from_predictions(
    predictions_df: pd.DataFrame,
    class_names: List[str],
    output_path: Optional[Path] = None,
    normalize: bool = True,
    show_inline: bool = True,
):
    """Generate and plot confusion matrix from predictions DataFrame using sklearn.

    Args:
        predictions_df: DataFrame with prediction results
        class_names: List of class names
        output_path: Optional path to save the plot. If None, plot is not saved.
        normalize: Whether to normalize the confusion matrix
        show_inline: Whether to display plot inline (for Jupyter notebooks)

    Returns:
        Tuple of (confusion_matrix, plot_statistics)
    """

    # Extract true and predicted labels
    y_true = predictions_df["true_label"].to_numpy()
    y_pred = predictions_df["predicted_label"].to_numpy()

    # Generate confusion matrix using sklearn
    conf_matrix = confusion_matrix(y_true, y_pred)

    logger.info(f"Generated confusion matrix with shape: {conf_matrix.shape}")
    logger.info(f"Total predictions: {len(y_true)}, Accuracy: {(y_true == y_pred).mean():.3f}")

    # Plot the confusion matrix using our updated function
    plot_stats = plot_confusion_matrix(
        conf_matrix, class_names, output_path, normalize, show_inline
    )

    return conf_matrix, plot_stats


def plot_data_augmentation_examples(
    dataset,
    class_names: List[str],
    output_path: Optional[Path] = None,
    num_examples: int = 5,
    show_inline: bool = True,
):
    """Plot examples of data augmentation effects using get_transforms.

    Args:
        dataset: PyTorch dataset to sample from
        class_names: List of class names
        output_path: Optional path to save the plot. If None, plot is not saved.
        num_examples: Number of augmentation examples to show
        show_inline: Whether to display plot inline (for Jupyter notebooks)

    Returns:
        Dict with augmentation information
    """

    # Get a sample image for demonstration
    sample_image, sample_label = None, None
    for i, (image, label) in enumerate(dataset):
        if i < 100:  # Look through first 100 samples
            sample_image, sample_label = image, label
            break

    if sample_image is None or sample_label is None:
        logger.warning("No sample image found for augmentation demo")
        return {"error": "No sample image found"}

    # Get original image (convert to PIL/numpy if needed)
    if isinstance(sample_image, torch.Tensor):
        if len(sample_image.shape) == 3:
            original_image = sample_image.squeeze().numpy()
        else:
            original_image = sample_image.numpy()
    else:
        original_image = np.array(sample_image)

    # Get transforms for training (with augmentation) and testing (without)
    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)

    # Generate augmented versions
    fig, axes = plt.subplots(2, num_examples + 1, figsize=(15, 6))

    # Original image (top row, first column)
    axes[0, 0].imshow(original_image, cmap="gray")
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    # Test transform (bottom row, first column)
    test_transformed = test_transform(original_image)
    if isinstance(test_transformed, torch.Tensor):
        test_img = test_transformed.squeeze().numpy()
        # Denormalize if needed
        if test_img.min() < 0:
            test_img = (test_img + 1) / 2
    else:
        test_img = test_transformed

    axes[1, 0].imshow(test_img, cmap="gray")
    axes[1, 0].set_title("Test Transform\n(No Augmentation)")
    axes[1, 0].axis("off")

    # Generate augmented examples
    for i in range(num_examples):
        # Apply training transform (with augmentation)
        augmented = train_transform(original_image)

        if isinstance(augmented, torch.Tensor):
            aug_img = augmented.squeeze().numpy()
            # Denormalize if needed
            if aug_img.min() < 0:
                aug_img = (aug_img + 1) / 2
        else:
            aug_img = augmented

        axes[0, i + 1].imshow(aug_img, cmap="gray")
        axes[0, i + 1].set_title(f"Augmented {i + 1}")
        axes[0, i + 1].axis("off")

        # Bottom row: show same test transform for comparison
        axes[1, i + 1].imshow(test_img, cmap="gray")
        axes[1, i + 1].set_title("Test Transform")
        axes[1, i + 1].axis("off")

    plt.suptitle(f"Data Augmentation Examples - {class_names[sample_label]}", fontsize=16)
    plt.tight_layout()

    # Save to file if output_path provided
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Data augmentation examples saved to {output_path}")

    # Show inline for Jupyter notebooks
    if show_inline:
        plt.show()
    else:
        plt.close()

    # Return augmentation information
    return {
        "sample_class": class_names[sample_label],
        "sample_label": sample_label,
        "num_examples": num_examples,
        "transforms_applied": "rotation, horizontal_flip, normalization",
    }


@app.command()
def plot_dataset_overview(output_dir: Path = FIGURES_DIR):
    """Generate overview plots of the Fashion-MNIST dataset."""

    logger.info("Generating dataset overview plots...")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset info
    dataset_info_path = PROCESSED_DATA_DIR / "dataset_info.pkl"
    if not dataset_info_path.exists():
        logger.info("Dataset info not found. Preparing datasets...")
        # Use imported prepare_datasets function
        train_dataset, val_dataset, test_dataset = prepare_datasets(
            RAW_DATA_DIR, PROCESSED_DATA_DIR, 0.1, 42
        )

        # Load dataset info after preparation
        with open(dataset_info_path, "rb") as f:
            dataset_info = pickle.load(f)
    else:
        with open(dataset_info_path, "rb") as f:
            dataset_info = pickle.load(f)

        # Prepare datasets for sample images
        train_dataset, val_dataset, test_dataset = prepare_datasets(
            RAW_DATA_DIR, PROCESSED_DATA_DIR, 0.1, 42
        )

    # Plot class distribution
    plot_class_distribution(dataset_info, output_dir / "class_distribution.png")

    # Plot sample images using the prepared dataset
    plot_sample_images(test_dataset, dataset_info["class_names"], output_dir / "sample_images.png")

    # Plot data augmentation examples using get_transforms
    plot_data_augmentation_examples(
        test_dataset, dataset_info["class_names"], output_dir / "data_augmentation_examples.png"
    )

    logger.success("Dataset overview plots generated successfully!")


@app.command()
def plot_training_results(model_type: str = "standard", output_dir: Path = FIGURES_DIR):
    """Generate training result plots."""

    logger.info(f"Generating training result plots for {model_type} model...")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training history
    history_path = MODELS_DIR / f"training_history_fashion_mnist_{model_type}.pkl"
    if not history_path.exists():
        logger.error(f"Training history not found at {history_path}")
        return

    with open(history_path, "rb") as f:
        history = pickle.load(f)

    # Plot training history
    plot_training_history(history, output_dir / f"training_history_{model_type}.png")

    logger.success("Training result plots generated successfully!")


@app.command()
def plot_evaluation_results(model_type: str = "standard", output_dir: Path = FIGURES_DIR):
    """Generate evaluation result plots."""

    logger.info(f"Generating evaluation plots for {model_type} model...")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load evaluation results
    eval_path = PROCESSED_DATA_DIR / f"evaluation_results_{model_type}.pkl"
    if not eval_path.exists():
        logger.error(f"Evaluation results not found at {eval_path}")
        return

    with open(eval_path, "rb") as f:
        evaluation_results = pickle.load(f)

    # Load predictions
    predictions_path = PROCESSED_DATA_DIR / f"test_predictions_{model_type}.csv"
    if predictions_path.exists():
        predictions_df = pd.read_csv(predictions_path)
        # Generate confusion matrix from predictions using sklearn
        generate_confusion_matrix_from_predictions(
            predictions_df,
            evaluation_results["class_names"],
            output_dir / f"confusion_matrix_from_predictions_{model_type}.png",
        )
    else:
        predictions_df = None

    # Generate plots using existing confusion matrix
    plot_confusion_matrix(
        evaluation_results["confusion_matrix"],
        evaluation_results["class_names"],
        output_dir / f"confusion_matrix_{model_type}.png",
    )

    plot_class_performance(evaluation_results, output_dir / f"class_performance_{model_type}.png")

    if predictions_df is not None:
        plot_model_predictions(
            predictions_df,
            evaluation_results["class_names"],
            output_dir / f"model_predictions_{model_type}.png",
        )

    logger.success("Evaluation plots generated successfully!")


@app.command()
def main(model_type: str = "standard", output_dir: Path = FIGURES_DIR):
    """Generate all visualization plots for the Fashion-MNIST project."""

    logger.info("Generating all visualization plots...")

    # Generate dataset overview
    plot_dataset_overview(output_dir)

    # Generate training plots
    plot_training_results(model_type, output_dir)

    # Generate evaluation plots
    plot_evaluation_results(model_type, output_dir)

    logger.success("All visualization plots generated successfully!")


if __name__ == "__main__":
    app()
