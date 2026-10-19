
import math

import torch

from quantizers import quantize_tensor, dequantize_tensor


BITS = 8
GROUP_SIZES = (32, 64, 128)
METHODS = ("affine", "symmetric")


def check_case(
    values,
    method,
    granularity,
    axis=0,
    group_size=64,
):
    result = quantize_tensor(
        values,
        bits=BITS,
        method=method,
        granularity=granularity,
        axis=axis,
        group_size=group_size,
    )
    restored = dequantize_tensor(result)

    # Shape, finite values, and legal integer codes.
    assert result.codes.shape == values.shape
    assert restored.shape == values.shape
    assert torch.isfinite(restored).all()
    assert result.codes.min().item() >= result.qmin
    assert result.codes.max().item() <= result.qmax

    # Verify how many independent parameter sets were created.
    if granularity == "per_tensor" or values.ndim < 2:
        expected_sets = 1
    elif granularity == "per_channel":
        expected_sets = values.shape[axis % values.ndim]
    elif granularity == "per_group":
        expected_sets = math.ceil(values.numel() / group_size)
    else:
        raise AssertionError(f"Unexpected granularity: {granularity}")

    actual_sets = result.scale.numel()
    assert actual_sets == expected_sets, (
        f"{method=}, {granularity=}, {axis=}, {group_size=}: "
        f"expected {expected_sets} scales, got {actual_sets}"
    )

    # Confirm that dequantization produces a usable reconstruction.
    max_error = (
        values.double() - restored.double()
    ).abs().max().item()

    return max_error, actual_sets, result.granularity


def main():
    torch.manual_seed(42)

    # A non-square matrix makes axis 0 and axis 1 distinguishable.
    # 7 * 19 = 133 elements, so every group size leaves a partial group.
    values = torch.randn(7, 19) * 2

    cases = [
        ("per_tensor", None, None),
        ("per_channel", 0, None),
        ("per_channel", 1, None),
        ("per_group", None, 32),
        ("per_group", None, 64),
        ("per_group", None, 128),
    ]

    print(f"Input shape: {tuple(values.shape)}")
    print(f"Input elements: {values.numel()}\n")

    for method in METHODS:
        for granularity, axis, group_size in cases:
            actual_axis = 0 if axis is None else axis
            actual_group_size = (
                64 if group_size is None else group_size
            )

            max_error, parameter_sets, actual_granularity = check_case(
                values,
                method,
                granularity,
                axis=actual_axis,
                group_size=actual_group_size,
            )

            axis_label = (
                f"axis={axis}" if axis is not None else "axis=—"
            )
            group_label = (
                f"group={group_size}"
                if group_size is not None
                else "group=—"
            )

            print(
                f"{method:9s} | "
                f"{granularity:11s} | "
                f"{axis_label:7s} | "
                f"{group_label:9s} | "
                f"scales={parameter_sets:2d} | "
                f"max_abs_error={max_error:.8f}"
            )

            assert actual_granularity == granularity

    # Explicitly test that a vector falls back to per-tensor mode.
    vector = torch.randn(19)

    result = quantize_tensor(
        vector,
        bits=BITS,
        method="affine",
        granularity="per_channel",
        axis=0,
    )

    assert result.granularity == "per_tensor"
    assert result.scale.numel() == 1
    assert dequantize_tensor(result).shape == vector.shape

    print("\nVector fallback test passed.")

    # Invalid group sizes must be rejected.
    for invalid_size in (0, -1):
        try:
            quantize_tensor(
                values,
                bits=BITS,
                method="affine",
                granularity="per_group",
                group_size=invalid_size,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"group_size={invalid_size} should raise ValueError"
            )

    print("Invalid group-size tests passed.")
    print("\nAll granularity tests passed.")


if __name__ == "__main__":
    main()
