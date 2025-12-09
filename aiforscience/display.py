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


def print_tokenization(text, tokenizer):
    """
    Print detailed tokenization information.

    Args:
        text: Input text string
        tokenizer: Hugging Face tokenizer
    """
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    print(f"Text: '{text}'")
    print(f"{'─'*50}")

    # Show token-by-token breakdown
    print("Tokens:")
    for i, (tok, tok_id) in enumerate(zip(tokens, token_ids)):
        # Clean up token representation
        display_tok = tok.replace('Ġ', '▁')  # GPT-2 uses Ġ for space
        print(f"  [{i:2d}] '{display_tok}' → ID: {tok_id}")

    print(f"{'─'*50}")
    print(f"Total tokens: {len(tokens)}")


def print_model_summary(model, title="Model Summary"):
    """
    Print a summary of a PyTorch model's parameters.

    Args:
        model: PyTorch model (nn.Module)
        title: Header title
    """
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")

    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        n_params = param.numel()
        total_params += n_params
        if param.requires_grad:
            trainable_params += n_params

    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Show breakdown by layer type
    param_by_type = {}
    for name, param in model.named_parameters():
        # Extract layer type from name
        parts = name.split('.')
        layer_type = parts[0] if len(parts) > 0 else 'other'
        if layer_type not in param_by_type:
            param_by_type[layer_type] = 0
        param_by_type[layer_type] += param.numel()

    print(f"\n  Parameters by component:")
    for layer_type, count in param_by_type.items():
        pct = 100 * count / total_params
        print(f"    {layer_type}: {count:,} ({pct:.1f}%)")

    print(f"\n{'='*50}\n")
