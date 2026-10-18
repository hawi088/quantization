
import torch

from quantizers import quantize_tensor, dequantize_tensor
from representations import convert_float


def quantize_state_dict(
    state_dict: dict[str, torch.Tensor],
    representation: str,
    granularity: str = "per_tensor",
    group_size: int = 64,
    channel_axis: int = 0,
) -> tuple[dict[str, torch.Tensor], dict]:
    """
    Quantize a state dict and return reconstructed tensors plus metrics.

    Supported representations:
        FP16, BF16,
        INT8_SYM, INT8_AFFINE,
        INT4_SYM, INT4_AFFINE,
        INT2_SYM, INT2_AFFINE.

    Integer quantization supports per_tensor, per_channel, and
    per_group granularities.
    """
    supported = {
        "FP16", "BF16",
        "INT8_SYM", "INT8_AFFINE",
        "INT4_SYM", "INT4_AFFINE",
        "INT2_SYM", "INT2_AFFINE",
    }

    if representation not in supported:
        raise ValueError(f"Unsupported representation: {representation}")

    if granularity not in ("per_tensor", "per_channel", "per_group"):
        raise ValueError(
            "granularity must be 'per_tensor', "
            "'per_channel', or 'per_group'"
        )

    if group_size <= 0:
        raise ValueError("group_size must be positive")

    reconstructed = {}
    total_squared_error = 0.0
    total_squared_weight = 0.0
    total_absolute_error = 0.0
    total_elements = 0
    estimated_bytes = 0
    quantized_tensor_count = 0
    parameter_set_count = 0

    for name, original in state_dict.items():
        if not original.is_floating_point():
            reconstructed[name] = original.clone()
            estimated_bytes += (
                original.numel() * original.element_size()
            )
            continue

        if representation in ("FP16", "BF16"):
            converted = convert_float(
                original, representation.lower()
            )
            restored = converted.to(dtype=original.dtype)
            estimated_bytes += original.numel() * 2

        else:
            method = (
                "symmetric"
                if representation.endswith("_SYM")
                else "affine"
            )
            bits = int(representation[3])

            result = quantize_tensor(
                original,
                bits=bits,
                method=method,
                granularity=granularity,
                axis=channel_axis,
                group_size=group_size,
            )

            restored = dequantize_tensor(result).to(
                dtype=original.dtype
            )

            # Estimated packed code storage. Actual serialization and
            # padding will be measured in the artifact stage.
            estimated_bytes += (
                original.numel() * bits + 7
            ) // 8

            if result.granularity == "per_tensor":
                parameter_sets = 1
            elif result.granularity == "per_channel":
                parameter_sets = original.shape[channel_axis % original.ndim]
            else:
                parameter_sets = (
                    original.numel() + group_size - 1
                ) // group_size

            parameter_set_count += parameter_sets
            estimated_bytes += parameter_sets * 4  # float32 scales

            if method == "affine":
                # Store each zero point as int32 in this estimate.
                estimated_bytes += parameter_sets * 4

            quantized_tensor_count += 1

        reconstructed[name] = restored

        difference = original.double() - restored.double()
        total_squared_error += difference.square().sum().item()
        total_squared_weight += original.double().square().sum().item()
        total_absolute_error += difference.abs().sum().item()
        total_elements += original.numel()

    metrics = {
        "estimated_weight_bytes": estimated_bytes,
        "weight_mae": (
            total_absolute_error / max(total_elements, 1)
        ),
        "weight_rmse": (
            total_squared_error / max(total_elements, 1)
        ) ** 0.5,
        "relative_weight_rmse": (
            (total_squared_error / total_squared_weight) ** 0.5
            if total_squared_weight > 0
            else 0.0
        ),
        "granularity": granularity,
        "group_size": group_size if granularity == "per_group" else None,
        "quantized_tensor_count": quantized_tensor_count,
        "parameter_set_count": parameter_set_count,
    }

    return reconstructed, metrics
