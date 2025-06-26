from pathlib import Path
import pickle
from typing import Any, List, Optional, Tuple

from loguru import logger
import numpy as np
import torch
from torch.utils.data import (  # random_split: alternative splitting method
    DataLoader,
    Dataset,
)
from torchvision import datasets, transforms
from tqdm import tqdm
import typer

from cnnFashionMnist.config import (
    AUGMENTATION_CONFIG,
    DATASET_CONFIG,
    DEVICE_CONFIG,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TRAINING_CONFIG,
)

app = typer.Typer()


class FashionMNISTDataset(Dataset):
    """Custom Dataset class for Fashion-MNIST data with transforms."""

    def __init__(self, data, targets, transform: Optional[transforms.Compose] = None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, target


def get_transforms(train: bool = True):
    """Get data transforms for training and validation."""
    if train:
        aug_config = AUGMENTATION_CONFIG["train"]
        transform_list: List[Any] = [transforms.ToPILImage()]

        if "random_rotation" in aug_config:
            transform_list.append(transforms.RandomRotation(aug_config["random_rotation"]))
        if "random_horizontal_flip" in aug_config:
            transform_list.append(
                transforms.RandomHorizontalFlip(aug_config["random_horizontal_flip"])
            )

        transform_list.append(transforms.ToTensor())

        if aug_config.get("normalize", True):
            transform_list.append(
                transforms.Normalize(DATASET_CONFIG["mean"], DATASET_CONFIG["std"])
            )

        return transforms.Compose(transform_list)
    else:
        aug_config = AUGMENTATION_CONFIG["val_test"]
        transform_list: List[Any] = [transforms.ToPILImage(), transforms.ToTensor()]

        if aug_config.get("normalize", True):
            transform_list.append(
                transforms.Normalize(DATASET_CONFIG["mean"], DATASET_CONFIG["std"])
            )

        return transforms.Compose(transform_list)


def download_fashion_mnist(data_dir: Path) -> Tuple[datasets.FashionMNIST, datasets.FashionMNIST]:
    """Download Fashion-MNIST dataset."""
    logger.info("Downloading Fashion-MNIST dataset...")

    with tqdm(total=2, desc="Downloading datasets") as pbar:
        train_dataset = datasets.FashionMNIST(
            root=data_dir, train=True, download=True, transform=None
        )
        pbar.update(1)

        test_dataset = datasets.FashionMNIST(
            root=data_dir, train=False, download=True, transform=None
        )
        pbar.update(1)

    logger.success("Fashion-MNIST dataset downloaded successfully.")
    return train_dataset, test_dataset


def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = TRAINING_CONFIG["batch_size"],
    num_workers: int = DEVICE_CONFIG["num_workers"],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training, validation, and testing."""

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,  # 64 samples per batch
        shuffle=True,  # ⭐ IMPORTANT: Randomize order
        num_workers=num_workers,  # 2 parallel workers
        pin_memory=DEVICE_CONFIG["pin_memory"],  # GPU optimization
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,  # Same batch size
        shuffle=False,  # ⭐ No shuffling for validation
        num_workers=num_workers,  # Same parallel workers
        pin_memory=DEVICE_CONFIG["pin_memory"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,  # Same batch size
        shuffle=False,  # ⭐ No shuffling for testing
        num_workers=num_workers,  # Same parallel workers
        pin_memory=DEVICE_CONFIG["pin_memory"],
    )

    return train_loader, val_loader, test_loader


def prepare_datasets(
    raw_data_dir: Path,
    processed_data_dir: Path,
    val_split: float = TRAINING_CONFIG["val_split"],
    random_seed: int = TRAINING_CONFIG["random_seed"],
) -> Tuple[Dataset, Dataset, Dataset]:
    """Prepare Fashion-MNIST datasets with train/validation/test splits."""

    # Download raw data
    train_raw, test_raw = download_fashion_mnist(raw_data_dir)

    # Convert to numpy arrays
    train_data = train_raw.data.numpy()  # [60000, 28, 28] - images
    train_targets = train_raw.targets.numpy()  # [60000] - labels
    test_data = test_raw.data.numpy()  # [10000, 28, 28] - images
    test_targets = test_raw.targets.numpy()  # [10000] - labels

    # Log dataset statistics using numpy
    logger.info(f"Training samples: {len(train_data)}, Test samples: {len(test_data)}")
    logger.info(f"Image shape: {train_data.shape[1:]}, Data type: {train_data.dtype}")
    logger.info(f"Data range: [{np.min(train_data)}, {np.max(train_data)}]")

    # Create train/validation split
    torch.manual_seed(random_seed)  # Set seed for reproducibility
    dataset_size = len(train_data)  # 60,000
    val_size = int(val_split * dataset_size)  # 0.1 * 60,000 = 6,000
    train_size = dataset_size - val_size  # 60,000 - 6,000 = 54,000

    indices = torch.randperm(dataset_size)  # Random shuffle of [0, 1, 2, ..., 59999]
    train_indices = indices[:train_size]  # First 54,000 indices
    val_indices = indices[train_size:]  # Last 6,000 indices

    # Split data
    train_data_split = train_data[train_indices]  # [54000, 28, 28]
    train_targets_split = train_targets[train_indices]  # [54000]
    val_data_split = train_data[val_indices]  # [6000, 28, 28]
    val_targets_split = train_targets[val_indices]  # [6000]

    # Create datasets with transforms
    train_dataset = FashionMNISTDataset(
        train_data_split, train_targets_split, get_transforms(train=True)
    )
    val_dataset = FashionMNISTDataset(
        val_data_split, val_targets_split, get_transforms(train=False)
    )
    test_dataset = FashionMNISTDataset(test_data, test_targets, get_transforms(train=False))

    # Save processed datasets
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    datasets_info = {
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "num_classes": 10,
        "class_names": [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ],
    }

    with open(processed_data_dir / "dataset_info.pkl", "wb") as f:
        pickle.dump(datasets_info, f)

    logger.info(
        f"Dataset prepared - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset


@app.command()
def main(
    batch_size: int = TRAINING_CONFIG["batch_size"],
    val_split: float = TRAINING_CONFIG["val_split"],
    random_seed: int = TRAINING_CONFIG["random_seed"],
    num_workers: int = DEVICE_CONFIG["num_workers"],
):
    """Prepare Fashion-MNIST dataset for training."""
    logger.info("Preparing Fashion-MNIST dataset...")

    # Create directories
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        RAW_DATA_DIR, PROCESSED_DATA_DIR, val_split, random_seed
    )

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size, num_workers
    )

    # Save data loaders info
    loaders_info = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "test_batches": len(test_loader),
    }

    with open(PROCESSED_DATA_DIR / "loaders_info.pkl", "wb") as f:
        pickle.dump(loaders_info, f)

    logger.success("Fashion-MNIST dataset preparation complete.")
    logger.info(
        f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}"
    )


if __name__ == "__main__":
    app()
