"""AI for Sciences - Utility Package for Practicals"""

from .visualization import (
    plot_gradient_descent_1d,
    plot_loss_history,
    plot_predictions,
    plot_gradient_step,
    plot_nonlinear_data,
)

from .display import (
    print_model_params,
    print_device_comparison,
)

from .data import (
    generate_linear_data,
    generate_nonlinear_data,
)

__all__ = [
    # Visualization
    "plot_gradient_descent_1d",
    "plot_loss_history",
    "plot_predictions",
    "plot_gradient_step",
    "plot_nonlinear_data",
    # Display
    "print_model_params",
    "print_device_comparison",
    # Data
    "generate_linear_data",
    "generate_nonlinear_data",
]
