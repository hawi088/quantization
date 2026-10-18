
import torch


def get_integer_range(
    bits: int,
    symmetric: bool = False,
) -> tuple[int, int]:
    """
    Return the signed integer code range for a given bit width.

    Affine:
        Uses all 2**bits signed codes.

    Symmetric:
        Uses equal positive and negative magnitudes,
        reserving one code compared with the full signed range.
    """
    if bits not in (2, 4, 8):
        raise ValueError("Supported integer bit widths are 2, 4, and 8.")

    if symmetric:
        qmax = (2 ** (bits - 1)) - 1
        return -qmax, qmax

    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1
    return qmin, qmax


def convert_float(
    values: torch.Tensor,
    precision: str,
) -> torch.Tensor:
    """
    Convert floating-point tensors to FP16 or BF16.

    This is floating-point precision conversion, not integer
    quantization. Values are not mapped using a scale or zero point.
    """
    if not values.is_floating_point():
        raise TypeError("values must have a floating-point dtype")

    formats = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    if precision not in formats:
        raise ValueError("precision must be 'fp16' or 'bf16'")

    return values.to(dtype=formats[precision])
