"""Visualization utilities for AI for Sciences practicals."""

import numpy as np
import matplotlib.pyplot as plt


def plot_gradient_descent_1d(f, theta_history, theta_range=(-5, 10), title="Gradient Descent"):
    """
    Plot the gradient descent trajectory on a 1D function.

    Args:
        f: Function to minimize (callable)
        theta_history: List of theta values during optimization
        theta_range: Tuple (min, max) for plotting range
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left plot: Function with trajectory
    theta_plot = np.linspace(theta_range[0], theta_range[1], 200)
    y_plot = [f(t) for t in theta_plot]

    axes[0].plot(theta_plot, y_plot, 'b-', linewidth=2, label='f(θ)')

    # Plot trajectory
    for i, theta in enumerate(theta_history):
        color = plt.cm.Reds(0.3 + 0.7 * i / len(theta_history))
        axes[0].scatter(theta, f(theta), c=[color], s=100, zorder=5)
        if i > 0:
            axes[0].annotate('', xy=(theta, f(theta)),
                           xytext=(theta_history[i-1], f(theta_history[i-1])),
                           arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    axes[0].scatter(theta_history[0], f(theta_history[0]), c='green', s=150,
                   marker='o', zorder=6, label='Start')
    axes[0].scatter(theta_history[-1], f(theta_history[-1]), c='red', s=150,
                   marker='*', zorder=6, label='End')

    axes[0].set_xlabel('θ', fontsize=12)
    axes[0].set_ylabel('f(θ)', fontsize=12)
    axes[0].set_title(title, fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right plot: Function value over iterations
    f_history = [f(t) for t in theta_history]
    axes[1].plot(range(len(f_history)), f_history, 'b-o', markersize=4)
    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('f(θ)', fontsize=12)
    axes[1].set_title('f(θ) over iterations', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_loss_history(losses, title="Training Loss"):
    """
    Plot training loss over epochs.

    Args:
        losses: List of loss values
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(range(1, len(losses) + 1), losses, 'b-', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add start and end annotations
    ax.scatter([1], [losses[0]], c='green', s=100, zorder=5, label=f'Start: {losses[0]:.4f}')
    ax.scatter([len(losses)], [losses[-1]], c='red', s=100, zorder=5, label=f'End: {losses[-1]:.4f}')
    ax.legend()

    plt.tight_layout()
    return fig


def plot_predictions(X, y, y_pred, title="Model Predictions"):
    """
    Plot true values vs predictions for 1D regression.

    Args:
        X: Input features (1D)
        y: True values
        y_pred: Predicted values
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Sort for line plot
    if X.ndim > 1:
        X_plot = X[:, 0]
    else:
        X_plot = X

    sort_idx = np.argsort(X_plot.flatten())

    ax.scatter(X_plot, y, c='blue', alpha=0.6, label='True values', s=50)
    ax.plot(X_plot.flatten()[sort_idx], y_pred.flatten()[sort_idx],
            'r-', linewidth=2, label='Predictions')

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_gradient_step(theta_before, theta_after, gradient, learning_rate, f, theta_range=(-5, 10)):
    """
    Visualize a single gradient descent step.

    Args:
        theta_before: Parameter value before update
        theta_after: Parameter value after update
        gradient: Gradient value
        learning_rate: Learning rate used
        f: Function being optimized
        theta_range: Range for plotting
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    theta_plot = np.linspace(theta_range[0], theta_range[1], 200)
    y_plot = [f(t) for t in theta_plot]

    ax.plot(theta_plot, y_plot, 'b-', linewidth=2, label='f(θ)')

    # Before point
    ax.scatter(theta_before, f(theta_before), c='green', s=200, zorder=5,
              label=f'Before: θ={theta_before:.3f}')

    # After point
    ax.scatter(theta_after, f(theta_after), c='red', s=200, zorder=5,
              label=f'After: θ={theta_after:.3f}')

    # Arrow showing update
    ax.annotate('', xy=(theta_after, f(theta_after)),
               xytext=(theta_before, f(theta_before)),
               arrowprops=dict(arrowstyle='->', color='purple', lw=3))

    # Tangent line at before point (showing gradient direction)
    tangent_x = np.linspace(theta_before - 1, theta_before + 1, 50)
    tangent_y = f(theta_before) + gradient * (tangent_x - theta_before)
    ax.plot(tangent_x, tangent_y, 'g--', linewidth=2, alpha=0.7, label=f'Tangent (gradient={gradient:.3f})')

    ax.set_xlabel('θ', fontsize=12)
    ax.set_ylabel('f(θ)', fontsize=12)
    ax.set_title(f'Gradient Step: θ_new = θ - lr × gradient = {theta_before:.3f} - {learning_rate} × {gradient:.3f} = {theta_after:.3f}',
                fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_nonlinear_data(X, y, y_pred=None, title="Data"):
    """
    Plot 2D non-linear data with optional predictions.

    Args:
        X: Input features (n_samples, 2)
        y: True values
        y_pred: Optional predicted values
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='coolwarm', alpha=0.7, s=50)
    plt.colorbar(scatter, ax=ax, label='y value')

    ax.set_xlabel('X₁', fontsize=12)
    ax.set_ylabel('X₂', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
