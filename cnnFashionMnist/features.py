from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
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
from cnnFashionMnist.modeling.predict import load_model

app = typer.Typer()

###### File Purpose
###### features.py handles:
######
###### 1. Feature extraction from CNN layers (trained & untrained models)
###### 2. Statistical analysis of learned representations
###### 3. Dimensionality reduction (PCA, t-SNE) for visualization
###### 4. Clustering analysis to understand feature groupings
###### 5. Feature visualization with professional plots
###### 6. Comparative analysis between different layers and models


class FeatureExtractor:
    """Extract features from CNN models for analysis."""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.features = {}  # Store extracted features
        self.hooks = []  # Store registered hooks

    def register_hooks(self, layer_names: Optional[List[str]] = None):
        """Register forward hooks to extract intermediate features."""
        if layer_names is None:
            # Default: extract from key layers
            layer_names = ["conv1", "conv3", "conv5", "fc1"]

        def hook_fn(name):
            def hook(module, input, output):
                self.features[name] = output.detach()

            return hook

        # Register hooks for specified layers
        for name, module in self.model.named_modules():
            if any(layer in name for layer in layer_names):
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)
                logger.info(f"Registered hook for layer: {name}")

    def extract_features(self, data_loader):
        """Extract features from all layers for given data."""
        self.model.eval()
        all_features: Dict[str, List[np.ndarray]] = {name: [] for name in self.features.keys()}
        all_labels: List[np.ndarray] = []
        all_predictions: List[np.ndarray] = []

        with torch.no_grad():
            for data, labels in tqdm(data_loader, desc="Extracting features"):
                data = data.to(self.device)

                # Forward pass (triggers hooks)
                outputs = self.model(data)
                predictions = torch.argmax(outputs, dim=1)

                # Store features from each layer
                for layer_name, features in self.features.items():
                    # Flatten features for analysis
                    flat_features = features.view(features.size(0), -1)
                    all_features[layer_name].append(flat_features.cpu().numpy())

                all_labels.append(labels.numpy())
                all_predictions.append(predictions.cpu().numpy())

        # Concatenate all batches
        concatenated_features: Dict[str, np.ndarray] = {}
        for layer_name in all_features:
            concatenated_features[layer_name] = np.vstack(all_features[layer_name])

        concatenated_labels = np.concatenate(all_labels)
        concatenated_predictions = np.concatenate(all_predictions)

        return concatenated_features, concatenated_labels, concatenated_predictions

    def cleanup(self):
        """Remove registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def analyze_feature_statistics(
    features: np.ndarray, labels: np.ndarray, class_names: List[str]
) -> Dict[str, Any]:
    """Analyze statistical properties of extracted features."""

    stats = {
        "feature_shape": features.shape,
        "mean_activation": np.mean(features),  # Average activation value
        "std_activation": np.std(features),  # Activation variability
        "sparsity": np.mean(features == 0),  # % of zero activations (ReLU effect)
        "per_class_stats": {},
    }

    # Per-class statistics
    for class_idx, class_name in enumerate(class_names):
        class_mask = labels == class_idx
        class_features = features[class_mask]

        if len(class_features) > 0:
            stats["per_class_stats"][class_name] = {
                "mean": np.mean(class_features),
                "std": np.std(class_features),
                "sparsity": np.mean(class_features == 0),
                "sample_count": len(class_features),
            }

    return stats


def create_feature_dataframe(
    features: np.ndarray, labels: np.ndarray, predictions: np.ndarray, class_names: List[str]
) -> pd.DataFrame:
    """Create a pandas DataFrame for easier feature analysis."""

    # Create DataFrame with basic info
    df = pd.DataFrame(
        {
            "true_label": labels,
            "predicted_label": predictions,
            "true_class": [class_names[i] for i in labels],
            "predicted_class": [class_names[i] for i in predictions],
            "correct_prediction": labels == predictions,
        }
    )

    # Add feature statistics
    df["feature_mean"] = np.mean(features, axis=1)
    df["feature_std"] = np.std(features, axis=1)
    df["feature_max"] = np.max(features, axis=1)
    df["feature_min"] = np.min(features, axis=1)
    df["feature_sparsity"] = np.mean(features == 0, axis=1)

    logger.info(f"Created feature DataFrame with shape: {df.shape}")
    return df


def perform_dimensionality_reduction(
    features: np.ndarray, labels: np.ndarray, method: str = "pca", n_components: int = 2
) -> Tuple[np.ndarray, Any]:
    """Perform dimensionality reduction on features."""

    logger.info(f"Performing {method.upper()} with {n_components} components...")

    if method.lower() == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
    elif method.lower() == "tsne":
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
    else:
        raise ValueError(f"Unknown method: {method}")

    reduced_features = reducer.fit_transform(features)

    return reduced_features, reducer


def cluster_features(features: np.ndarray, n_clusters: int = 10) -> Tuple[np.ndarray, KMeans]:
    """Perform K-means clustering on features."""

    logger.info(f"Performing K-means clustering with {n_clusters} clusters...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)

    return cluster_labels, kmeans


def plot_feature_distribution(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    layer_name: str,
    output_path: Optional[Path] = None,
    show_inline: bool = True,
):
    """Plot distribution of feature activations.

    Args:
        features: Feature matrix (samples x features)
        labels: Class labels for each sample
        class_names: List of class names
        layer_name: Name of the layer being analyzed
        output_path: Optional path to save the plot. If None, plot is not saved.
        show_inline: Whether to display plot inline (for Jupyter notebooks)

    Returns:
        Dict with feature distribution statistics
    """

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Overall distribution
    axes[0, 0].hist(features.flatten(), bins=50, alpha=0.7, density=True)
    axes[0, 0].set_title(f"{layer_name} - Overall Activation Distribution")
    axes[0, 0].set_xlabel("Activation Value")
    axes[0, 0].set_ylabel("Density")

    # Mean activation per class
    class_means = []
    for class_idx in range(len(class_names)):
        class_mask = labels == class_idx
        if np.any(class_mask):
            class_means.append(np.mean(features[class_mask]))
        else:
            class_means.append(0)

    axes[0, 1].bar(range(len(class_names)), class_means)
    axes[0, 1].set_title(f"{layer_name} - Mean Activation per Class")
    axes[0, 1].set_xlabel("Class")
    axes[0, 1].set_ylabel("Mean Activation")
    axes[0, 1].set_xticks(range(len(class_names)))
    axes[0, 1].set_xticklabels(class_names, rotation=45, ha="right")

    # Feature variance
    feature_vars = np.var(features, axis=0)
    axes[1, 0].hist(feature_vars, bins=50, alpha=0.7)
    axes[1, 0].set_title(f"{layer_name} - Feature Variance Distribution")
    axes[1, 0].set_xlabel("Variance")
    axes[1, 0].set_ylabel("Count")

    # Sparsity per class
    class_sparsity = []
    for class_idx in range(len(class_names)):
        class_mask = labels == class_idx
        if np.any(class_mask):
            class_sparsity.append(np.mean(features[class_mask] == 0))
        else:
            class_sparsity.append(0)

    axes[1, 1].bar(range(len(class_names)), class_sparsity)
    axes[1, 1].set_title(f"{layer_name} - Sparsity per Class")
    axes[1, 1].set_xlabel("Class")
    axes[1, 1].set_ylabel("Sparsity (% zeros)")
    axes[1, 1].set_xticks(range(len(class_names)))
    axes[1, 1].set_xticklabels(class_names, rotation=45, ha="right")

    plt.tight_layout()

    # Save to file if output_path provided
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Feature distribution plot saved to {output_path}")

    # Show inline for Jupyter notebooks
    if show_inline:
        plt.show()
    else:
        plt.close()

    # Return feature statistics
    return {
        "layer_name": layer_name,
        "total_features": features.shape[1],
        "total_samples": features.shape[0],
        "overall_mean": np.mean(features),
        "overall_std": np.std(features),
        "overall_sparsity": np.mean(features == 0),
        "class_means": dict(zip(class_names, class_means)),
        "class_sparsity": dict(zip(class_names, class_sparsity)),
        "feature_variance_mean": np.mean(feature_vars),
        "feature_variance_std": np.std(feature_vars),
    }


def plot_dimensionality_reduction(
    reduced_features: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    method: str,
    layer_name: str,
    output_path: Optional[Path] = None,
    show_inline: bool = True,
):
    """Plot dimensionality reduction results.

    Args:
        reduced_features: 2D reduced feature matrix (samples x 2)
        labels: Class labels for each sample
        class_names: List of class names
        method: Dimensionality reduction method (e.g., 'pca', 'tsne')
        layer_name: Name of the layer being analyzed
        output_path: Optional path to save the plot. If None, plot is not saved.
        show_inline: Whether to display plot inline (for Jupyter notebooks)

    Returns:
        Dict with dimensionality reduction statistics
    """

    plt.figure(figsize=(12, 10))

    # Create color map
    colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, len(class_names)))

    class_separations = []
    for class_idx, (class_name, color) in enumerate(zip(class_names, colors)):
        class_mask = labels == class_idx
        class_points = reduced_features[class_mask]

        plt.scatter(
            class_points[:, 0],
            class_points[:, 1],
            c=[color],
            label=class_name,
            alpha=0.6,
            s=20,
        )

        # Calculate intra-class variance for separation metric
        if len(class_points) > 1:
            class_var = np.var(class_points, axis=0).sum()
            class_separations.append(class_var)

    plt.title(f"{method.upper()} Visualization - {layer_name}")
    plt.xlabel(f"{method.upper()} Component 1")
    plt.ylabel(f"{method.upper()} Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save to file if output_path provided
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"{method.upper()} plot saved to {output_path}")

    # Show inline for Jupyter notebooks
    if show_inline:
        plt.show()
    else:
        plt.close()

    # Return dimensionality reduction statistics
    return {
        "method": method.upper(),
        "layer_name": layer_name,
        "num_samples": reduced_features.shape[0],
        "num_classes": len(class_names),
        "component_1_range": [
            float(reduced_features[:, 0].min()),
            float(reduced_features[:, 0].max()),
        ],
        "component_2_range": [
            float(reduced_features[:, 1].min()),
            float(reduced_features[:, 1].max()),
        ],
        "avg_intra_class_variance": np.mean(class_separations) if class_separations else 0,
    }


def plot_feature_correlation_matrix(
    features: np.ndarray,
    layer_name: str,
    output_path: Optional[Path] = None,
    max_features: int = 50,
    show_inline: bool = True,
):
    """Plot correlation matrix of features using seaborn.

    Args:
        features: Feature matrix (samples x features)
        layer_name: Name of the layer being analyzed
        output_path: Optional path to save the plot. If None, plot is not saved.
        max_features: Maximum number of features to include in correlation analysis
        show_inline: Whether to display plot inline (for Jupyter notebooks)

    Returns:
        Dict with correlation matrix statistics
    """

    # Use subset of features if too many
    if features.shape[1] > max_features:
        feature_subset = features[:, :max_features]
        logger.info(f"Using subset of {max_features} features for correlation analysis")
    else:
        feature_subset = features

    # Create DataFrame for easier handling
    feature_df = pd.DataFrame(
        feature_subset, columns=[f"feat_{i}" for i in range(feature_subset.shape[1])]
    )

    # Calculate correlation matrix
    corr_matrix = feature_df.corr()

    # Create the plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap="coolwarm",
        center=0,
        square=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.8},
    )

    plt.title(f"Feature Correlation Matrix - {layer_name}")
    plt.tight_layout()

    # Save to file if output_path provided
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Feature correlation matrix saved to {output_path}")

    # Show inline for Jupyter notebooks
    if show_inline:
        plt.show()
    else:
        plt.close()

    # Return correlation statistics
    return {
        "layer_name": layer_name,
        "features_analyzed": feature_subset.shape[1],
        "total_features": features.shape[1],
        "avg_correlation": float(np.mean(np.abs(corr_matrix.values))),
        "max_correlation": float(
            np.max(np.abs(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]))
        ),
        "min_correlation": float(
            np.min(np.abs(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]))
        ),
        "highly_correlated_pairs": int(
            np.sum(np.abs(corr_matrix.values) > 0.8) - len(corr_matrix)
        ),  # exclude diagonal
    }


def plot_class_feature_comparison(
    df: pd.DataFrame, layer_name: str, output_path: Optional[Path] = None, show_inline: bool = True
):
    """Plot feature statistics comparison across classes using seaborn.

    Args:
        df: DataFrame with feature statistics and class information
        layer_name: Name of the layer being analyzed
        output_path: Optional path to save the plot. If None, plot is not saved.
        show_inline: Whether to display plot inline (for Jupyter notebooks)

    Returns:
        Dict with class comparison statistics
    """

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Feature mean by class
    sns.boxplot(data=df, x="true_class", y="feature_mean", ax=axes[0, 0])
    axes[0, 0].set_title(f"{layer_name} - Feature Mean by Class")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # Feature std by class
    sns.boxplot(data=df, x="true_class", y="feature_std", ax=axes[0, 1])
    axes[0, 1].set_title(f"{layer_name} - Feature Std by Class")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # Feature sparsity by class
    sns.boxplot(data=df, x="true_class", y="feature_sparsity", ax=axes[1, 0])
    axes[1, 0].set_title(f"{layer_name} - Feature Sparsity by Class")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # Prediction accuracy by feature mean
    sns.scatterplot(
        data=df,
        x="feature_mean",
        y="correct_prediction",
        hue="true_class",
        alpha=0.6,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title(f"{layer_name} - Prediction vs Feature Mean")
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    # Save to file if output_path provided
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Class feature comparison plot saved to {output_path}")

    # Show inline for Jupyter notebooks
    if show_inline:
        plt.show()
    else:
        plt.close()

    # Calculate comparison statistics
    class_stats = {}
    for class_name in df["true_class"].unique():
        class_data = df[df["true_class"] == class_name]
        class_stats[class_name] = {
            "mean_feature_mean": float(class_data["feature_mean"].mean()),
            "mean_feature_std": float(class_data["feature_std"].mean()),
            "mean_sparsity": float(class_data["feature_sparsity"].mean()),
            "prediction_accuracy": float(class_data["correct_prediction"].mean())
            if "correct_prediction" in class_data.columns
            else 0,
        }

    return {
        "layer_name": layer_name,
        "num_samples": len(df),
        "num_classes": len(df["true_class"].unique()),
        "overall_feature_mean": float(df["feature_mean"].mean()),
        "overall_feature_std": float(df["feature_std"].mean()),
        "overall_sparsity": float(df["feature_sparsity"].mean()),
        "class_statistics": class_stats,
    }


@app.command()
def extract_cnn_features(
    model_path: Path = MODELS_DIR / FILE_PATTERNS["model"].format(model_type="standard"),
    model_type: str = "standard",
    layer_names: Optional[List[str]] = None,
    batch_size: int = 64,
    num_samples: int = 1000,
    save_features: bool = True,
):
    """Extract features from trained CNN model."""

    logger.info("Starting CNN feature extraction...")

    # Setup device
    device = DEVICE_CONFIG["device"]

    # Load model
    model, checkpoint = load_model(model_path, model_type, device)

    # Prepare data
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        RAW_DATA_DIR, PROCESSED_DATA_DIR, 0.1, 42
    )

    # Use subset of test data for feature extraction
    try:
        dataset_size = len(test_dataset)  # type: ignore
    except (TypeError, AttributeError):
        dataset_size = num_samples
    if num_samples < dataset_size:
        subset_indices = np.random.choice(dataset_size, num_samples, replace=False)
        test_subset = torch.utils.data.Subset(test_dataset, subset_indices.tolist())
    else:
        test_subset = test_dataset

    # Create data loaders using the imported function
    # For feature extraction, we only need the test loader, so we'll create dummy train/val
    dummy_dataset = test_subset  # Use same dataset as placeholder
    _, _, test_loader = create_data_loaders(
        dummy_dataset, dummy_dataset, test_subset, batch_size, 2
    )

    # Extract features
    extractor = FeatureExtractor(model, device)
    extractor.register_hooks(layer_names)

    features_dict, labels, predictions = extractor.extract_features(test_loader)
    extractor.cleanup()

    class_names = DATASET_CONFIG["class_names"]

    # Analyze and save features
    results = {
        "features": features_dict,
        "labels": labels,
        "predictions": predictions,
        "class_names": class_names,
        "model_type": model_type,
        "statistics": {},
    }

    for layer_name, features in features_dict.items():
        logger.info(f"Analyzing layer: {layer_name}")

        # Statistical analysis
        stats = analyze_feature_statistics(features, labels, class_names)
        results["statistics"][layer_name] = stats

        # Create DataFrame for pandas-based analysis
        feature_df = create_feature_dataframe(features, labels, predictions, class_names)
        results[f"dataframe_{layer_name}"] = feature_df

        logger.info(
            f"Layer {layer_name}: Shape={stats['feature_shape']}, "
            f"Mean={stats['mean_activation']:.3f}, "
            f"Sparsity={stats['sparsity']:.3f}"
        )

    # Save results
    if save_features:
        features_path = PROCESSED_DATA_DIR / f"cnn_features_{model_type}.pkl"
        with open(features_path, "wb") as f:
            pickle.dump(results, f)
        logger.info(f"Features saved to {features_path}")

    logger.success("Feature extraction completed!")


@app.command()
def extract_features_from_untrained_model(
    model_type: str = "standard",
    layer_names: Optional[List[str]] = None,
    batch_size: int = 64,
    num_samples: int = 1000,
    save_features: bool = True,
):
    """Extract features from untrained CNN model (for comparison with trained features)."""

    logger.info("Starting feature extraction from untrained model...")

    # Setup device
    device = DEVICE_CONFIG["device"]

    # Create untrained model using the imported function
    model = create_model(model_type=model_type, num_classes=10)
    model.to(device)
    model.eval()
    logger.info(f"Created untrained {model_type} model")

    # Prepare data
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        RAW_DATA_DIR, PROCESSED_DATA_DIR, 0.1, 42
    )

    # Use subset of test data for feature extraction
    try:
        dataset_size = len(test_dataset)  # type: ignore
    except (TypeError, AttributeError):
        dataset_size = num_samples
    if num_samples < dataset_size:
        subset_indices = np.random.choice(dataset_size, num_samples, replace=False)
        test_subset = torch.utils.data.Subset(test_dataset, subset_indices.tolist())
    else:
        test_subset = test_dataset

    # Create data loaders using the imported function
    dummy_dataset = test_subset  # Use same dataset as placeholder
    _, _, test_loader = create_data_loaders(
        dummy_dataset, dummy_dataset, test_subset, batch_size, 2
    )

    # Extract features
    extractor = FeatureExtractor(model, device)
    extractor.register_hooks(layer_names)

    features_dict, labels, predictions = extractor.extract_features(test_loader)
    extractor.cleanup()

    class_names = DATASET_CONFIG["class_names"]

    # Analyze and save features
    results = {
        "features": features_dict,
        "labels": labels,
        "predictions": predictions,
        "class_names": class_names,
        "model_type": f"{model_type}_untrained",
        "statistics": {},
    }

    for layer_name, features in features_dict.items():
        logger.info(f"Analyzing layer: {layer_name}")

        # Statistical analysis
        stats = analyze_feature_statistics(features, labels, class_names)
        results["statistics"][layer_name] = stats

        # Create DataFrame for pandas-based analysis
        feature_df = create_feature_dataframe(features, labels, predictions, class_names)
        results[f"dataframe_{layer_name}"] = feature_df

        logger.info(
            f"Layer {layer_name}: Shape={stats['feature_shape']}, "
            f"Mean={stats['mean_activation']:.3f}, "
            f"Sparsity={stats['sparsity']:.3f}"
        )

    # Save results
    if save_features:
        features_path = PROCESSED_DATA_DIR / f"cnn_features_{model_type}_untrained.pkl"
        with open(features_path, "wb") as f:
            pickle.dump(results, f)
        logger.info(f"Untrained model features saved to {features_path}")

    logger.success("Untrained model feature extraction completed!")


@app.command()
def visualize_features(
    model_type: str = "standard",
    layer_name: str = "fc1",
    reduction_method: str = "pca",
    n_components: int = 2,
):
    """Visualize extracted CNN features."""

    logger.info("Starting feature visualization...")

    # Load extracted features
    features_path = PROCESSED_DATA_DIR / f"cnn_features_{model_type}.pkl"
    if not features_path.exists():
        logger.error("Features not found. Run extract-cnn-features first.")
        return

    with open(features_path, "rb") as f:
        results = pickle.load(f)

    if layer_name not in results["features"]:
        logger.error(
            f"Layer {layer_name} not found. Available: {list(results['features'].keys())}"
        )
        return

    features = results["features"][layer_name]
    labels = results["labels"]
    class_names = results["class_names"]

    # Create output directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Plot feature distributions
    dist_path = FIGURES_DIR / f"feature_distribution_{layer_name}_{model_type}.png"
    plot_feature_distribution(features, labels, class_names, layer_name, dist_path)

    # Plot correlation matrix using seaborn
    corr_path = FIGURES_DIR / f"feature_correlation_{layer_name}_{model_type}.png"
    plot_feature_correlation_matrix(features, layer_name, corr_path)

    # Create DataFrame and plot class comparisons using seaborn
    feature_df = create_feature_dataframe(
        features, labels, labels, class_names
    )  # Using labels as predictions for viz
    comparison_path = FIGURES_DIR / f"class_feature_comparison_{layer_name}_{model_type}.png"
    plot_class_feature_comparison(feature_df, layer_name, comparison_path)

    # Dimensionality reduction and visualization
    reduced_features, reducer = perform_dimensionality_reduction(
        features, labels, reduction_method, n_components
    )

    viz_path = FIGURES_DIR / f"feature_{reduction_method}_{layer_name}_{model_type}.png"
    plot_dimensionality_reduction(
        reduced_features, labels, class_names, reduction_method, layer_name, viz_path
    )

    logger.success("Feature visualization completed!")


@app.command()
def analyze_feature_clusters(
    model_type: str = "standard", layer_name: str = "fc1", n_clusters: int = 10
):
    """Analyze feature clusters and compare with true labels."""

    logger.info("Starting cluster analysis...")

    # Load extracted features
    features_path = PROCESSED_DATA_DIR / f"cnn_features_{model_type}.pkl"
    if not features_path.exists():
        logger.error("Features not found. Run extract-cnn-features first.")
        return

    with open(features_path, "rb") as f:
        results = pickle.load(f)

    features = results["features"][layer_name]
    labels = results["labels"]
    class_names = results["class_names"]

    # Perform clustering
    cluster_labels, kmeans = cluster_features(features, n_clusters)

    # Analyze cluster purity
    cluster_analysis = {}
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_true_labels = labels[cluster_mask]

        if len(cluster_true_labels) > 0:
            unique_labels, counts = np.unique(cluster_true_labels, return_counts=True)
            dominant_label = unique_labels[np.argmax(counts)]
            purity = np.max(counts) / len(cluster_true_labels)

            cluster_analysis[cluster_id] = {
                "size": len(cluster_true_labels),
                "dominant_class": class_names[dominant_label],
                "purity": purity,
                "label_distribution": dict(zip(unique_labels, counts)),
            }

    # Log results
    logger.info("Cluster Analysis Results:")
    for cluster_id, analysis in cluster_analysis.items():
        logger.info(
            f"Cluster {cluster_id}: Size={analysis['size']}, "
            f"Dominant={analysis['dominant_class']}, "
            f"Purity={analysis['purity']:.3f}"
        )

    # Save analysis
    analysis_path = PROCESSED_DATA_DIR / f"cluster_analysis_{layer_name}_{model_type}.pkl"
    with open(analysis_path, "wb") as f:
        pickle.dump(
            {
                "cluster_analysis": cluster_analysis,
                "cluster_labels": cluster_labels,
                "kmeans": kmeans,
                "layer_name": layer_name,
                "model_type": model_type,
            },
            f,
        )

    logger.success(f"Cluster analysis saved to {analysis_path}")


@app.command()
def main(model_type: str = "standard", num_samples: int = 1000, visualization: bool = True):
    """Extract and analyze CNN features for Fashion-MNIST."""

    logger.info("Starting complete feature analysis pipeline...")

    # Step 1: Extract features
    extract_cnn_features(model_type=model_type, num_samples=num_samples, save_features=True)

    # Step 2: Visualize features (if requested)
    if visualization:
        for layer in ["conv1", "conv3", "fc1"]:
            try:
                visualize_features(model_type=model_type, layer_name=layer)
            except Exception as e:
                logger.warning(f"Could not visualize layer {layer}: {e}")

        # Step 3: Cluster analysis
        try:
            analyze_feature_clusters(model_type=model_type)
        except Exception as e:
            logger.warning(f"Could not perform cluster analysis: {e}")

    logger.success("Complete feature analysis pipeline finished!")


if __name__ == "__main__":
    app()
