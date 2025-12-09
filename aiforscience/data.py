"""Data generation utilities for practicals."""

import numpy as np
import torch


def generate_linear_data(n_samples=100, n_features=1, noise=1.0, bias=5.0, weight_scale=3.0, seed=42):
    """
    Generate synthetic data for linear regression.

    Args:
        n_samples: Number of samples
        n_features: Number of input features
        noise: Standard deviation of Gaussian noise
        bias: Bias value (larger = harder to learn initially)
        weight_scale: Scale of the weights
        seed: Random seed for reproducibility

    Returns:
        X: Input tensor of shape (n_samples, n_features)
        y: Target tensor of shape (n_samples, 1)
        true_weights: True weights used for generation
        true_bias: True bias used for generation
    """
    np.random.seed(seed)

    # Generate true parameters with larger values
    true_weights = np.random.randn(n_features, 1) * weight_scale
    true_bias = np.array([bias])

    # Generate input data
    X = np.random.randn(n_samples, n_features) * 2

    # Generate targets with noise
    y = X @ true_weights + true_bias + np.random.randn(n_samples, 1) * noise

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    print(f"Generated linear dataset:")
    print(f"  - Samples: {n_samples}")
    print(f"  - Features: {n_features}")
    print(f"  - X shape: {X_tensor.shape}")
    print(f"  - y shape: {y_tensor.shape}")
    print(f"  - True weights: {true_weights.flatten()}")
    print(f"  - True bias: {true_bias[0]:.4f}")

    return X_tensor, y_tensor, true_weights, true_bias


def generate_nonlinear_data(n_samples=200, noise=0.3, seed=42):
    """
    Generate synthetic non-linear data (spiral/circular pattern).
    This data CANNOT be solved by a linear model.

    Args:
        n_samples: Number of samples
        noise: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        X: Input tensor of shape (n_samples, 2)
        y: Target tensor of shape (n_samples, 1)
    """
    np.random.seed(seed)

    # Generate circular/radial pattern
    # y = sin(||x||) which creates concentric rings
    X = np.random.randn(n_samples, 2) * 2

    # Non-linear target: based on distance from origin + angle
    r = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    angle = np.arctan2(X[:, 1], X[:, 0])

    # Create a complex non-linear pattern
    y = np.sin(2 * r) + 0.5 * np.cos(3 * angle) + np.random.randn(n_samples) * noise
    y = y.reshape(-1, 1)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    print(f"Generated non-linear dataset:")
    print(f"  - Samples: {n_samples}")
    print(f"  - Features: 2")
    print(f"  - X shape: {X_tensor.shape}")
    print(f"  - y shape: {y_tensor.shape}")
    print(f"  - Pattern: y = sin(2*r) + 0.5*cos(3*angle) + noise")
    print(f"  - This data CANNOT be solved by a linear model!")

    return X_tensor, y_tensor
