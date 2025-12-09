"""Display utilities for printing training information."""

import torch


def print_model_params(model, title="Model Parameters"):
    """
    Print model parameters in a formatted way.

    Args:
        model: PyTorch model
        title: Header title
    """
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")

    total_params = 0
    for name, param in model.named_parameters():
        print(f"\n  {name}:")
        print(f"    Shape: {list(param.shape)}")
        print(f"    Values: {param.data.flatten()[:5].tolist()}")
        print(f"    Requires grad: {param.requires_grad}")
        total_params += param.numel()

    print(f"\n  Total parameters: {total_params}")
    print(f"{'='*50}\n")


def print_device_comparison(cpu_time, gpu_time, speedup=None):
    """
    Print comparison between CPU and GPU training times.

    Args:
        cpu_time: Training time on CPU (seconds)
        gpu_time: Training time on GPU (seconds)
        speedup: Optional pre-calculated speedup
    """
    if speedup is None:
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')

    print(f"\n{'='*50}")
    print(" CPU vs GPU Performance Comparison")
    print(f"{'='*50}")
    print(f"\n  CPU Time:  {cpu_time:.4f} seconds")
    print(f"  GPU Time:  {gpu_time:.4f} seconds")
    print(f"\n  Speedup:   {speedup:.2f}x faster on GPU")

    # Visual bar
    max_bar = 40
    cpu_bar = int(max_bar)
    gpu_bar = int(max_bar * gpu_time / cpu_time) if cpu_time > 0 else max_bar

    print(f"\n  CPU: [{'█' * cpu_bar}]")
    print(f"  GPU: [{'█' * gpu_bar}{'░' * (cpu_bar - gpu_bar)}]")

    print(f"\n{'='*50}\n")
