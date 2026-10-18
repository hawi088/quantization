
import torch


def quantize_symmetric(
    values: torch.Tensor,
    qmin: int,
    qmax: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize floating-point values using symmetric min-max quantization.

    The zero point is fixed at zero.

    Returns:
        q: Integer quantization codes.
        scale: Positive scalar quantization scale.

    Reconstruction:
        reconstructed = scale * q
    """
    if not values.is_floating_point():
        raise TypeError("values must have a floating-point dtype")

    if values.numel() == 0:
        raise ValueError("values must not be empty")

    if qmin >= qmax:
        raise ValueError("qmin must be smaller than qmax")

    if qmin != -qmax:
        raise ValueError(
            "Symmetric quantization requires qmin == -qmax"
        )

    if not torch.isfinite(values).all():
        raise ValueError("values must contain only finite numbers")

    max_abs = values.abs().max().to(torch.float64)

    # All-zero tensors need no quantization error and would otherwise
    # produce a zero scale.
    if max_abs.item() == 0.0:
        scale = torch.tensor(
            1.0, dtype=torch.float64, device=values.device
        )
        q = torch.zeros_like(values, dtype=torch.int64)
        return q, scale

    scale = max_abs / qmax

    q = torch.round(values.to(torch.float64) / scale)
    q = q.clamp(qmin, qmax).to(torch.int64)

    return q, scale


def dequantize_symmetric(
    q: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct approximate values:
        reconstructed = scale * q
    """
    if scale.numel() != 1 or not torch.isfinite(scale).all():
        raise ValueError("scale must be a finite scalar")

    if scale.item() <= 0:
        raise ValueError("scale must be positive")

    return scale * q.to(torch.float64)
