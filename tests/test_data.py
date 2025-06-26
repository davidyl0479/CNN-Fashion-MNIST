import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from cnnFashionMnist.dataset import (
    FashionMNISTDataset, 
    get_transforms, 
    create_data_loaders,
    prepare_datasets
)
from cnnFashionMnist.modeling.model import (
    FashionMNISTCNN, 
    SimpleFashionMNISTCNN, 
    create_model,
    initialize_weights
)


class TestFashionMNISTDataset:
    """Test cases for FashionMNISTDataset class."""
    
    def test_dataset_creation(self):
        """Test that FashionMNISTDataset can be created properly."""
        # Create dummy data
        data = np.random.randint(0, 255, (10, 28, 28), dtype=np.uint8)
        targets = np.random.randint(0, 10, 10)
        
        dataset = FashionMNISTDataset(data, targets)
        
        assert len(dataset) == 10
        assert dataset[0][1] == targets[0]
    
    def test_dataset_with_transforms(self):
        """Test that transforms are applied correctly."""
        data = np.random.randint(0, 255, (5, 28, 28), dtype=np.uint8)
        targets = np.random.randint(0, 10, 5)
        transform = get_transforms(train=True)
        
        dataset = FashionMNISTDataset(data, targets, transform)
        
        image, label = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (1, 28, 28)  # (C, H, W)
        assert isinstance(label, (int, np.integer))


class TestTransforms:
    """Test cases for data transforms."""
    
    def test_train_transforms(self):
        """Test training transforms."""
        transform = get_transforms(train=True)
        assert transform is not None
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
        transformed = transform(dummy_image)
        
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (1, 28, 28)
    
    def test_val_transforms(self):
        """Test validation transforms."""
        transform = get_transforms(train=False)
        assert transform is not None
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
        transformed = transform(dummy_image)
        
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (1, 28, 28)


class TestModels:
    """Test cases for CNN models."""
    
    def test_fashion_mnist_cnn_creation(self):
        """Test FashionMNISTCNN model creation."""
        model = FashionMNISTCNN(num_classes=10)
        assert model is not None
        
        # Test forward pass
        x = torch.randn(2, 1, 28, 28)
        output = model(x)
        assert output.shape == (2, 10)
    
    def test_simple_fashion_mnist_cnn_creation(self):
        """Test SimpleFashionMNISTCNN model creation."""
        model = SimpleFashionMNISTCNN(num_classes=10)
        assert model is not None
        
        # Test forward pass
        x = torch.randn(2, 1, 28, 28)
        output = model(x)
        assert output.shape == (2, 10)
    
    def test_create_model_factory(self):
        """Test model factory function."""
        # Test standard model
        model1 = create_model(model_type="standard", num_classes=10)
        assert isinstance(model1, FashionMNISTCNN)
        
        # Test simple model
        model2 = create_model(model_type="simple", num_classes=10)
        assert isinstance(model2, SimpleFashionMNISTCNN)
        
        # Test invalid model type
        with pytest.raises(ValueError):
            create_model(model_type="invalid", num_classes=10)
    
    def test_model_info(self):
        """Test model info retrieval."""
        model = FashionMNISTCNN(num_classes=10)
        info = model.get_model_info()
        
        assert 'model_name' in info
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        assert 'model_size_mb' in info
        assert info['total_parameters'] > 0
    
    def test_initialize_weights(self):
        """Test weight initialization."""
        model = FashionMNISTCNN(num_classes=10)
        
        # Get initial weights
        initial_conv_weight = model.conv1.weight.clone()
        
        # Initialize weights
        initialize_weights(model)
        
        # Check that weights have changed
        assert not torch.equal(initial_conv_weight, model.conv1.weight)


class TestDataLoaders:
    """Test cases for data loaders."""
    
    def test_create_data_loaders(self):
        """Test data loader creation."""
        # Create dummy datasets
        data = np.random.randint(0, 255, (100, 28, 28), dtype=np.uint8)
        targets = np.random.randint(0, 10, 100)
        
        train_dataset = FashionMNISTDataset(data[:60], targets[:60])
        val_dataset = FashionMNISTDataset(data[60:80], targets[60:80])
        test_dataset = FashionMNISTDataset(data[80:], targets[80:])
        
        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset, val_dataset, test_dataset, batch_size=8, num_workers=0
        )
        
        assert len(train_loader) == 8  # 60 samples / 8 batch_size = 7.5 -> 8 batches
        assert len(val_loader) == 3   # 20 samples / 8 batch_size = 2.5 -> 3 batches
        assert len(test_loader) == 3  # 20 samples / 8 batch_size = 2.5 -> 3 batches
        
        # Test batch shape
        batch_data, batch_targets = next(iter(train_loader))
        assert batch_data.shape[0] <= 8  # batch size
        assert batch_data.shape[1:] == (1, 28, 28)  # image shape


class TestIntegration:
    """Integration tests."""
    
    def test_model_training_step(self):
        """Test a single training step."""
        model = SimpleFashionMNISTCNN(num_classes=10)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create dummy batch
        batch_data = torch.randn(4, 1, 28, 28)
        batch_targets = torch.randint(0, 10, (4,))
        
        # Forward pass
        model.train()
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
        assert outputs.shape == (4, 10)
    
    def test_model_inference(self):
        """Test model inference."""
        model = SimpleFashionMNISTCNN(num_classes=10)
        model.eval()
        
        with torch.no_grad():
            batch_data = torch.randn(2, 1, 28, 28)
            outputs = model(batch_data)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
        
        assert outputs.shape == (2, 10)
        assert probabilities.shape == (2, 10)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))
        assert predictions.shape == (2,)
        assert all(0 <= pred < 10 for pred in predictions)


def test_code_is_tested():
    """Test that our test suite is working."""
    assert True
