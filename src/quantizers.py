
from dataclasses import dataclass
from typing import Union

import torch

from quantization_math import quantize_affine, dequantize_affine
from symmetric_quantization import (
    quantize_symmetric,
    dequantize_symmetric,
)
from representations import get_integer_range , convert_float

#we specified this because zero point can be either an int or a tensor, depending on the quantization method and granularity. 
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


def _quantize_rows(
    rows: torch.Tensor,
    bits: int,
    method: str,
    valid_mask: torch.Tensor | None = None,
):
    """
    Quantize independent rows in parallel.

    rows has shape [number_of_slices, elements_per_slice].
    Each row gets independent parameters. valid_mask excludes padding
    from min/max calculations for a final, incomplete group.
    """
    qmin, qmax = get_integer_range(
        bits, symmetric=(method == "symmetric")
    )

    rows = rows.to(torch.float64)

    if valid_mask is None:
        valid_mask = torch.ones_like(rows, dtype=torch.bool)

    if method == "affine":
        row_min = rows.masked_fill(
            ~valid_mask, float("inf")
        ).amin(dim=1)
        row_max = rows.masked_fill(
            ~valid_mask, float("-inf")
        ).amax(dim=1)

        constant = row_min == row_max
        positive_constant = constant & (row_min > 0)
        negative_constant = constant & (row_min < 0)
        zero_constant = constant & (row_min == 0)
        varying = ~constant

        # Avoid division by zero while computing the ordinary branch.
        safe_range = torch.where(
            varying, row_max - row_min, torch.ones_like(row_min)
        )
        scale = safe_range / (qmax - qmin)

        # Apply the existing affine equations to nonconstant slices.
        zero_point = torch.round(qmin - row_min / scale)

        # Preserve the project's explicit constant-tensor handling.
        scale = torch.where(
            positive_constant,
            row_min / qmax,
            scale,
        )
        scale = torch.where(
            negative_constant,
            row_min / qmin,
            scale,
        )
        scale = torch.where(
            zero_constant, torch.ones_like(scale), scale
        )

        zero_point = torch.where(
            constant, torch.zeros_like(zero_point), zero_point
        )

        codes = torch.round(
            rows / scale.unsqueeze(1)
            + zero_point.unsqueeze(1)
        ).clamp(qmin, qmax)

        codes = torch.where(
            positive_constant.unsqueeze(1),
            torch.full_like(codes, qmax),
            codes,
        )
        codes = torch.where(
            negative_constant.unsqueeze(1),
            torch.full_like(codes, qmin),
            codes,
        )
        codes = torch.where(
            zero_constant.unsqueeze(1),
            torch.zeros_like(codes),
            codes,
        )

    else:
        # Symmetric quantization uses zero point 0.
        row_max_abs = rows.abs().masked_fill(
            ~valid_mask, 0.0
        ).amax(dim=1)

        scale = row_max_abs / qmax
        scale = torch.where(
            row_max_abs == 0,
            torch.ones_like(scale),
            scale,
        )

        zero_point = torch.zeros_like(scale)
        codes = torch.round(
            rows / scale.unsqueeze(1)
        ).clamp(qmin, qmax)

    return codes.to(torch.int64), scale, zero_point


def _make_result(
    codes,
    scales,
    zero_points,
    bits,
    method,
    granularity,
    axis,
    group_size,
    original_shape,
):
    qmin, qmax = get_integer_range(
        bits, symmetric=(method == "symmetric")
    )

    if method == "symmetric":
        zero_point = 0
    elif granularity == "per_tensor":
        zero_point = int(zero_points.reshape(-1)[0].item())
    else:
        zero_point = zero_points.to(torch.int64)

    return QuantizationResult(
        codes=codes,
        scale=scales,
        zero_point=zero_point,
        qmin=qmin,
        qmax=qmax,
        bits=bits,
        method=method,
        granularity=granularity,
        axis=axis,
        group_size=group_size,
        original_shape=tuple(original_shape),
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
    Quantize a floating-point tensor.

    per_tensor:
        One parameter set for the complete tensor.

    per_channel:
        One parameter set per slice along axis. For a 2-D matrix,
        axis=0 means rows and axis=1 means columns.

    per_group:
        Flatten the tensor and quantize consecutive groups of
        group_size elements independently.
    """
    if not values.is_floating_point():
        raise TypeError("values must have a floating-point dtype")

    if values.numel() == 0:
        raise ValueError("values must not be empty")

    if method not in ("affine", "symmetric"):
        raise ValueError("method must be 'affine' or 'symmetric'")

    if granularity not in ("per_tensor", "per_channel", "per_group"):
        raise ValueError(
            "granularity must be 'per_tensor', "
            "'per_channel', or 'per_group'"
        )

    if granularity == "per_group" and group_size <= 0:
        raise ValueError("group_size must be positive")

    original_shape = tuple(values.shape)

    if granularity == "per_tensor":
        if method == "affine":
            codes, scale, zero_point = quantize_affine(
                values, *get_integer_range(bits, symmetric=False)
            )
        else:
            codes, scale = quantize_symmetric(
                values, *get_integer_range(bits, symmetric=True)
            )
            zero_point = 0

        return QuantizationResult(
            codes=codes,
            scale=scale,
            zero_point=zero_point,
            qmin=get_integer_range(
                bits, symmetric=(method == "symmetric")
            )[0],
            qmax=get_integer_range(
                bits, symmetric=(method == "symmetric")
            )[1],
            bits=bits,
            method=method,
            granularity="per_tensor",
            axis=axis,
            group_size=group_size,
            original_shape=original_shape,
        )

    if granularity == "per_channel":
        if values.ndim < 2:
            # if the diamention is less than 2, there will be no channel so we apply per tensor 
            return quantize_tensor(
                values, bits, method, "per_tensor", axis, group_size
            )

        normalized_axis = axis % values.ndim

        # Move the selected axis to the front, then treat each slice
        # along that axis as one row for vectorized quantization.
        moved = values.movedim(normalized_axis, 0).contiguous()
        rows = moved.reshape(moved.shape[0], -1)

        codes, scales, zero_points = _quantize_rows(
            rows, bits, method
        )

        restored_codes = codes.reshape(moved.shape).movedim(
            0, normalized_axis
        ).contiguous()

        return _make_result(
            restored_codes,
            scales,
            zero_points,
            bits,
            method,
            "per_channel",
            normalized_axis,
            group_size,
            original_shape,
        )

    # Vectorized per-group quantization, including a partial final group.
    flat = values.reshape(-1).to(torch.float64)
    number_of_groups = (
        flat.numel() + group_size - 1
    ) // group_size
    padded_count = number_of_groups * group_size
    padding = padded_count - flat.numel()

    if padding:
        padded = torch.cat([
            flat,
            torch.zeros(padding, dtype=flat.dtype, device=flat.device),
        ])
    else:
        padded = flat

    rows = padded.reshape(number_of_groups, group_size)

    valid_mask = (
        torch.arange(padded_count, device=flat.device) < flat.numel()
    ).reshape(number_of_groups, group_size)

    codes, scales, zero_points = _quantize_rows(
        rows, bits, method, valid_mask
    )

    codes = codes.reshape(-1)[:flat.numel()].reshape(original_shape)

    return _make_result(
        codes,
        scales,
        zero_points,
        bits,
        method,
        "per_group",
        axis,
        group_size,
        original_shape,
    )


def dequantize_tensor(
    result: QuantizationResult,
) -> torch.Tensor:
    """Reconstruct floating-point values from quantized codes."""

    if result.granularity == "per_tensor":
        if result.method == "affine":
            return dequantize_affine(
                result.codes, result.scale, result.zero_point
            )
        if result.method == "symmetric":
            return dequantize_symmetric(
                result.codes, result.scale
            )

    if result.method not in ("affine", "symmetric"):
        raise ValueError(f"Unsupported method: {result.method}")

    if result.granularity == "per_channel":
        moved_codes = result.codes.movedim(
            result.axis, 0
        ).contiguous()
        rows = moved_codes.reshape(moved_codes.shape[0], -1).to(
            torch.float64
        )

        if result.method == "symmetric":
            reconstructed = rows * result.scale.unsqueeze(1)
        else:
            zero_points = result.zero_point.to(
                device=rows.device, dtype=torch.float64
            )
            reconstructed = (
                result.scale.unsqueeze(1)
                * (rows - zero_points.unsqueeze(1))
            )

        return reconstructed.reshape(moved_codes.shape).movedim(
            0, result.axis
        ).contiguous()

    if result.granularity == "per_group":
        flat_codes = result.codes.reshape(-1).to(torch.float64)
        group_index = (
            torch.arange(flat_codes.numel(), device=flat_codes.device)
            // result.group_size
        )
        scales = result.scale[group_index]

        if result.method == "symmetric":
            reconstructed = flat_codes * scales
        else:
            zero_points = result.zero_point.to(
                device=flat_codes.device, dtype=torch.float64
            )
            reconstructed = scales * (
                flat_codes - zero_points[group_index]
            )

        return reconstructed.reshape(result.original_shape)

    raise ValueError(
        f"Unsupported granularity: {result.granularity}"
    )
