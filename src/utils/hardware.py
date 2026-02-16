"""Hardware detection and VRAM estimation utilities."""

from __future__ import annotations


def get_gpu_info() -> dict | None:
    """Detect available GPU and return memory information.

    Returns:
        Dict with 'name', 'total_memory_mb', 'free_memory_mb' keys,
        or None if no GPU is available.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory / (1024**2)
        free = total - torch.cuda.memory_allocated(device) / (1024**2)

        return {
            "name": torch.cuda.get_device_name(device),
            "total_memory_mb": round(total, 1),
            "free_memory_mb": round(free, 1),
        }
    except ImportError:
        return None


def estimate_vram_requirement(
    model_params_b: float,
    quantization_bits: int = 4,
    adapter_overhead_pct: float = 0.05,
) -> float:
    """Estimate VRAM needed for training in MB.

    Args:
        model_params_b: Number of model parameters in billions.
        quantization_bits: Quantization level (4 or 8 bit).
        adapter_overhead_pct: Estimated adapter overhead as fraction.

    Returns:
        Estimated VRAM requirement in MB.
    """
    bytes_per_param = quantization_bits / 8
    base_memory_mb = model_params_b * 1e9 * bytes_per_param / (1024**2)
    adapter_memory_mb = base_memory_mb * adapter_overhead_pct

    # Rough estimate: optimizer states + gradients ~ 2x adapter memory
    training_overhead_mb = adapter_memory_mb * 2

    return round(base_memory_mb + adapter_memory_mb + training_overhead_mb, 1)
