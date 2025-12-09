"""Data generation utilities for practicals."""

import numpy as np
import torch


def generate_linear_data(n_samples=100, n_features=1, noise=0.3, seed=42):
    """
    Generate synthetic data for linear regression.

    Args:
        n_samples: Number of samples
        n_features: Number of input features
        noise: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        X: Input tensor of shape (n_samples, n_features)
        y: Target tensor of shape (n_samples, 1)
        true_weights: True weights used for generation
        true_bias: True bias used for generation
    """
    np.random.seed(seed)

    # Generate true parameters
    true_weights = np.random.randn(n_features, 1) * 2  # Random weights
    true_bias = np.random.randn(1) * 2  # Random bias

    # Generate input data
    X = np.random.randn(n_samples, n_features) * 2

    # Generate targets with noise
    y = X @ true_weights + true_bias + np.random.randn(n_samples, 1) * noise

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    print(f"Generated dataset:")
    print(f"  - Samples: {n_samples}")
    print(f"  - Features: {n_features}")
    print(f"  - X shape: {X_tensor.shape}")
    print(f"  - y shape: {y_tensor.shape}")
    print(f"  - True weights: {true_weights.flatten()}")
    print(f"  - True bias: {true_bias[0]:.4f}")

    return X_tensor, y_tensor, true_weights, true_bias
