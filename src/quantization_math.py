
import torch


def quantize_affine(
    values: torch.Tensor,
    qmin: int,
    qmax: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Quantize floating-point values using affine min-max quantization.

    Returns:
        q: Integer quantization codes.
        scale: Positive scalar quantization scale.
        zero_point: Integer zero point.

    Reconstruction:
        reconstructed = scale * (q - zero_point)
    """
    if not values.is_floating_point():
        raise TypeError("values must have a floating-point dtype")

    if values.numel() == 0:
        raise ValueError("values must not be empty")

    if qmin >= qmax:
        raise ValueError("qmin must be smaller than qmax")

    if not qmin <= 0 <= qmax:
        raise ValueError(
            "This affine implementation requires qmin <= 0 <= qmax"
        )

    if not torch.isfinite(values).all():
        raise ValueError("values must contain only finite numbers")

    r_min = values.min().to(torch.float64)
    r_max = values.max().to(torch.float64)

    # Handle constant tensors separately.
    if r_min.item() == r_max.item():
        constant = r_min.item()

        if constant > 0:
            scale = torch.tensor(
                constant / qmax,
                dtype=torch.float64,
                device=values.device,
            )
            zero_point = 0
            q = torch.full_like(values, qmax, dtype=torch.int64)

        elif constant < 0:
            scale = torch.tensor(
                constant / qmin,
                dtype=torch.float64,
                device=values.device,
            )
            zero_point = 0
            q = torch.full_like(values, qmin, dtype=torch.int64)

        else:
            scale = torch.tensor(
                1.0,
                dtype=torch.float64,
                device=values.device,
            )
            zero_point = 0
            q = torch.zeros_like(values, dtype=torch.int64)

        return q, scale, zero_point

    # Initial scale from the observed range.
    initial_scale = (r_max - r_min) / (qmax - qmin)

    # Calculate and clamp the zero point.
    zero_point = round(
        qmin - (r_min / initial_scale).item()
    )
    zero_point = max(qmin, min(qmax, zero_point))

    # Adjust the scale to ensure both endpoints fit in the code range.
    scale_candidates = []

    if r_max.item() > 0 and qmax > zero_point:
        scale_candidates.append(
            r_max / (qmax - zero_point)
        )

    if r_min.item() < 0 and zero_point > qmin:
        scale_candidates.append(
            -r_min / (zero_point - qmin)
        )

    if scale_candidates:
        scale = torch.stack(scale_candidates).max()
    else:
        scale = initial_scale

    # Quantize and clamp codes to the permitted integer range.
    q = torch.round(
        values.to(torch.float64) / scale + zero_point
    )
    q = q.clamp(qmin, qmax).to(torch.int64)

    return q, scale, zero_point


def dequantize_affine(
    q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: int,
) -> torch.Tensor:
    """
    Reconstruct approximate floating-point values:
        reconstructed = scale * (q - zero_point)
    """
    if scale.numel() != 1 or scale.item() <= 0:
        raise ValueError("scale must be a positive scalar")

    return scale * (q.to(torch.float64) - zero_point)
