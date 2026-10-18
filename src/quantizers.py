
from dataclasses import dataclass

import torch

from quantization_math import (
    quantize_affine,
    dequantize_affine,
)
from symmetric_quantization import (
    quantize_symmetric,
    dequantize_symmetric,
)
from representations import get_integer_range


@dataclass
class QuantizationResult:
    """Codes and metadata required to reconstruct a tensor."""

    codes: torch.Tensor
    scale: torch.Tensor
    zero_point: int
    qmin: int
    qmax: int
    bits: int
    method: str


def quantize_tensor(
    values: torch.Tensor,
    bits: int,
    method: str = "affine",
) -> QuantizationResult:
    """
    Quantize a tensor using the selected method.

    Supported methods:
        affine: learns scale and zero point from the tensor.
        symmetric: uses a zero point of zero.
    """
    if method not in ("affine", "symmetric"):
        raise ValueError(
            "method must be 'affine' or 'symmetric'"
        )

    qmin, qmax = get_integer_range(
        bits,
        symmetric=(method == "symmetric"),
    )

    if method == "affine":
        codes, scale, zero_point = quantize_affine(
            values, qmin, qmax
        )
    else:
        codes, scale = quantize_symmetric(
            values, qmin, qmax
        )
        zero_point = 0

    return QuantizationResult(
        codes=codes,
        scale=scale,
        zero_point=zero_point,
        qmin=qmin,
        qmax=qmax,
        bits=bits,
        method=method,
    )


def dequantize_tensor(
    result: QuantizationResult,
) -> torch.Tensor:
    """Reconstruct approximate floating-point values."""
    if result.method == "affine":
        return dequantize_affine(
            result.codes,
            result.scale,
            result.zero_point,
        )

    if result.method == "symmetric":
        return dequantize_symmetric(
            result.codes,
            result.scale,
        )

    raise ValueError(
        f"Unsupported quantization method: {result.method}"
    )
