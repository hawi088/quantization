
from dataclasses import dataclass
from typing import Union

import torch

from quantization_math import quantize_affine, dequantize_affine
from symmetric_quantization import (
    quantize_symmetric,
    dequantize_symmetric,
)
from representations import get_integer_range


ZeroPoint = Union[int, torch.Tensor]


@dataclass
class QuantizationResult:
    """Integer codes and metadata needed to reconstruct a tensor."""

    codes: torch.Tensor
    scale: torch.Tensor
    zero_point: ZeroPoint
    qmin: int
    qmax: int
    bits: int
    method: str
    granularity: str = "per_tensor"
    axis: int = 0
    group_size: int = 64
    original_shape: tuple = ()


def _quantize_one(
    values: torch.Tensor,
    bits: int,
    method: str,
) -> QuantizationResult:
    """Apply the existing scalar-parameter quantizer."""
    qmin, qmax = get_integer_range(
        bits,
        symmetric=(method == "symmetric"),
    )

    if method == "affine":
        codes, scale, zero_point = quantize_affine(
            values, qmin, qmax
        )
    else:
        codes, scale = quantize_symmetric(values, qmin, qmax)
        zero_point = 0

    return QuantizationResult(
        codes=codes,
        scale=scale,
        zero_point=zero_point,
        qmin=qmin,
        qmax=qmax,
        bits=bits,
        method=method,
        original_shape=tuple(values.shape),
    )


def quantize_tensor(
    values: torch.Tensor,
    bits: int,
    method: str = "affine",
    granularity: str = "per_tensor",
    axis: int = 0,
    group_size: int = 64,
) -> QuantizationResult:
    """
    Quantize a tensor per tensor, per channel, or per group.

    per_tensor:
        One scale and zero point for the entire tensor.

    per_channel:
        One parameter set per slice along axis. For a matrix of
        linear-layer weights, axis=0 means one set per output row.
        One-dimensional tensors fall back to per-tensor quantization.

    per_group:
        Flatten the tensor and quantize consecutive groups of
        group_size elements independently. The final group may be
        smaller than group_size.

    The affine equations are unchanged; they are applied separately
    to each tensor, channel, or group.
    """
    if not values.is_floating_point():
        raise TypeError("values must have a floating-point dtype")

    if method not in ("affine", "symmetric"):
        raise ValueError("method must be 'affine' or 'symmetric'")

    if granularity not in ("per_tensor", "per_channel", "per_group"):
        raise ValueError(
            "granularity must be 'per_tensor', "
            "'per_channel', or 'per_group'"
        )

    if values.numel() == 0:
        raise ValueError("values must not be empty")

    if granularity == "per_tensor":
        result = _quantize_one(values, bits, method)
        result.granularity = "per_tensor"
        return result

    if granularity == "per_channel":
        if values.ndim < 2:
            result = _quantize_one(values, bits, method)
            result.granularity = "per_tensor"
            return result

        normalized_axis = axis % values.ndim
        codes = torch.empty_like(values, dtype=torch.int64)
        scales = []
        zero_points = []

        for channel_index in range(values.shape[normalized_axis]):
            channel = values.select(normalized_axis, channel_index)
            channel_result = _quantize_one(channel, bits, method)

            codes.select(
                normalized_axis, channel_index
            ).copy_(channel_result.codes)

            scales.append(channel_result.scale.reshape(()))
            zero_points.append(
                int(channel_result.zero_point)
                if isinstance(channel_result.zero_point, int)
                else channel_result.zero_point.item()
            )

        return QuantizationResult(
            codes=codes,
            scale=torch.stack(scales),
            zero_point=torch.tensor(
                zero_points, dtype=torch.int64, device=values.device
            ),
            qmin=get_integer_range(
                bits, symmetric=(method == "symmetric")
            )[0],
            qmax=get_integer_range(
                bits, symmetric=(method == "symmetric")
            )[1],
            bits=bits,
            method=method,
            granularity="per_channel",
            axis=normalized_axis,
            group_size=group_size,
            original_shape=tuple(values.shape),
        )

    # Per-group quantization.
    if group_size <= 0:
        raise ValueError("group_size must be a positive integer")

    flat = values.reshape(-1)
    code_chunks = []
    scales = []
    zero_points = []

    for start in range(0, flat.numel(), group_size):
        chunk = flat[start:start + group_size]
        chunk_result = _quantize_one(chunk, bits, method)

        code_chunks.append(chunk_result.codes.reshape(-1))
        scales.append(chunk_result.scale.reshape(()))
        zero_points.append(
            int(chunk_result.zero_point)
            if isinstance(chunk_result.zero_point, int)
            else chunk_result.zero_point.item()
        )

    qmin, qmax = get_integer_range(
        bits, symmetric=(method == "symmetric")
    )

    return QuantizationResult(
        codes=torch.cat(code_chunks).reshape(values.shape),
        scale=torch.stack(scales),
        zero_point=torch.tensor(
            zero_points, dtype=torch.int64, device=values.device
        ),
        qmin=qmin,
        qmax=qmax,
        bits=bits,
        method=method,
        granularity="per_group",
        group_size=group_size,
        original_shape=tuple(values.shape),
    )


def dequantize_tensor(
    result: QuantizationResult,
) -> torch.Tensor:
    """Reconstruct floating-point values from integer codes."""

    def reconstruct(codes, scale, zero_point):
        if result.method == "affine":
            return dequantize_affine(codes, scale, zero_point)
        if result.method == "symmetric":
            return dequantize_symmetric(codes, scale)
        raise ValueError(
            f"Unsupported quantization method: {result.method}"
        )

    if result.granularity == "per_tensor":
        return reconstruct(
            result.codes, result.scale, result.zero_point
        )

    if result.granularity == "per_channel":
        output = torch.empty(
            result.original_shape,
            dtype=torch.float64,
            device=result.codes.device,
        )

        for index in range(result.codes.shape[result.axis]):
            zero_point = (
                result.zero_point[index]
                if isinstance(result.zero_point, torch.Tensor)
                else result.zero_point
            )
            reconstructed = reconstruct(
                result.codes.select(result.axis, index),
                result.scale[index],
                zero_point,
            )
            output.select(result.axis, index).copy_(reconstructed)

        return output

    if result.granularity == "per_group":
        flat_codes = result.codes.reshape(-1)
        output = torch.empty(
            flat_codes.shape,
            dtype=torch.float64,
            device=flat_codes.device,
        )

        for group_index, start in enumerate(
            range(0, flat_codes.numel(), result.group_size)
        ):
            end = min(start + result.group_size, flat_codes.numel())
            zero_point = (
                result.zero_point[group_index]
                if isinstance(result.zero_point, torch.Tensor)
                else result.zero_point
            )
            output[start:end] = reconstruct(
                flat_codes[start:end],
                result.scale[group_index],
                zero_point,
            )

        return output.reshape(result.original_shape)

    raise ValueError(
        f"Unsupported granularity: {result.granularity}"
    )
