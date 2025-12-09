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
        print(f"    Values: {param.data.numpy().flatten()[:5]}...")  # Show first 5 values
        print(f"    Requires grad: {param.requires_grad}")
        total_params += param.numel()

    print(f"\n  Total parameters: {total_params}")
    print(f"{'='*50}\n")


def print_training_step(epoch, loss, params_before=None, params_after=None, gradients=None):
    """
    Print detailed information about a training step.

    Args:
        epoch: Current epoch number
        loss: Loss value
        params_before: Dict of parameters before update
        params_after: Dict of parameters after update
        gradients: Dict of gradients
    """
    print(f"\n{'─'*40}")
    print(f" Epoch {epoch}")
    print(f"{'─'*40}")
    print(f"  Loss: {loss:.6f}")

    if gradients:
        print(f"\n  Gradients:")
        for name, grad in gradients.items():
            if grad is not None:
                print(f"    {name}: {grad.flatten()[:3].tolist()}...")

    if params_before and params_after:
        print(f"\n  Parameter Updates:")
        for name in params_before:
            before = params_before[name].flatten()[:3].tolist()
            after = params_after[name].flatten()[:3].tolist()
            print(f"    {name}:")
            print(f"      Before: {before}...")
            print(f"      After:  {after}...")


def print_gradient_info(model):
    """
    Print gradient information for all model parameters.

    Args:
        model: PyTorch model after backward pass
    """
    print(f"\n{'='*50}")
    print(" Gradient Information")
    print(f"{'='*50}")

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            print(f"\n  {name}:")
            print(f"    Gradient shape: {list(grad.shape)}")
            print(f"    Gradient mean: {grad.mean().item():.6f}")
            print(f"    Gradient std: {grad.std().item():.6f}")
            print(f"    Gradient min: {grad.min().item():.6f}")
            print(f"    Gradient max: {grad.max().item():.6f}")
        else:
            print(f"\n  {name}: No gradient computed")

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


def print_batch_info(batch_idx, total_batches, batch_loss, batch_size):
    """
    Print information about a training batch.

    Args:
        batch_idx: Current batch index
        total_batches: Total number of batches
        batch_loss: Loss for this batch
        batch_size: Size of the batch
    """
    progress = (batch_idx + 1) / total_batches * 100
    print(f"  Batch {batch_idx + 1}/{total_batches} ({progress:.0f}%) | "
          f"Batch size: {batch_size} | Loss: {batch_loss:.6f}")
