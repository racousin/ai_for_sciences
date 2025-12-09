"""AI for Sciences - Utility Package for Practicals"""

from .visualization import (
    plot_gradient_descent_1d,
    plot_loss_history,
    plot_predictions,
    plot_gradient_step,
)

from .display import (
    print_model_params,
    print_training_step,
    print_gradient_info,
    print_device_comparison,
)

from .data import (
    generate_linear_data,
)

__all__ = [
    # Visualization
    "plot_gradient_descent_1d",
    "plot_loss_history",
    "plot_predictions",
    "plot_gradient_step",
    # Display
    "print_model_params",
    "print_training_step",
    "print_gradient_info",
    "print_device_comparison",
    # Data
    "generate_linear_data",
]
