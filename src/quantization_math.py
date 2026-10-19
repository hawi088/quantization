
import torch


def quantize_affine(
    values: torch.Tensor,
    qmin: int,
    qmax: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Quantize floating-point values using the specified affine equations.

    scale = (r_max - r_min) / (qmax - qmin)
    zero_point = round(qmin - r_min / scale)
    q = clamp(round(values / scale + zero_point), qmin, qmax)

    Returns:
        q: Integer quantization codes.
        scale: Scalar quantization scale.
        zero_point: Integer zero point.
    """
    if not values.is_floating_point():
        raise TypeError("values must have a floating-point dtype")

    if values.numel() == 0: #check number of elements in the tensor
        raise ValueError("values must not be empty")

    if qmin >= qmax:
        raise ValueError("qmin must be smaller than qmax")

    if not qmin <= 0 <= qmax:
        raise ValueError("Require qmin <= 0 <= qmax")

    if not torch.isfinite(values).all():
        raise ValueError("values must contain only finite numbers")

    values32 = values.to(torch.float32)
    r_min = values32.min()
    r_max = values32.max()

    # if all elements in a tensor are the same, the resulting scale is zero which casuses division by zero
    if r_min.item() == r_max.item():
        constant = r_min.item()

        if constant == 0:
            scale = torch.tensor(
                1.0, dtype=torch.float32, device=values.device
            )
            zero_point = 0
            q = torch.zeros_like(values, dtype=torch.int64)
        else:
            # Represent the constant exactly using an endpoint code.
            q_code = qmax if constant > 0 else qmin
            scale = torch.tensor(
                constant / q_code,
                dtype=torch.float32,
                device=values.device,
            )
            zero_point = 0
            q = torch.full_like(
                values, q_code, dtype=torch.int64
            )

        return q, scale, zero_point

    # 1. Scale: use the formula
    scale = (r_max - r_min) / (qmax - qmin)

    # 2. Zero point: derive it from the minimum endpoint.
    zero_point = round(
        qmin - (r_min / scale).item()
    )

    # 3. Quantize, round, and clamp to the integer range.
    q = torch.round(values32 / scale + zero_point)
    q = q.clamp(qmin, qmax).to(torch.int64)

    return q, scale, zero_point


def dequantize_affine(
    q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: int,
) -> torch.Tensor:
    """
    Reconstruct values using:
        r_hat = scale * (q - zero_point)
    """
    if scale.numel() != 1 or not torch.isfinite(scale).all():
        raise ValueError("scale must be a finite scalar")

    if scale.item() <= 0:
        raise ValueError("scale must be positive")

    return scale * (q.to(torch.float64) - zero_point)
