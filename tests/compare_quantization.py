
import torch

from quantization_math import (
    quantize_affine,
    dequantize_affine,
)
from symmetric_quantization import (
    quantize_symmetric,
    dequantize_symmetric,
)


def compare_methods(
    values: torch.Tensor,
    bits: int,
    symmetric_qmax: int,
) -> None:
    # Use a signed integer range for the affine method.
    affine_qmin = -(2 ** (bits - 1))
    affine_qmax = (2 ** (bits - 1)) - 1

    # Symmetric quantization reserves equal positive and negative ranges.
    symmetric_qmin = -symmetric_qmax

    # Affine quantization.
    affine_codes, affine_scale, affine_zero_point = quantize_affine(
        values,
        affine_qmin,
        affine_qmax,
    )
    affine_reconstructed = dequantize_affine(
        affine_codes,
        affine_scale,
        affine_zero_point,
    )

    # Symmetric quantization.
    symmetric_codes, symmetric_scale = quantize_symmetric(
        values,
        symmetric_qmin,
        symmetric_qmax,
    )
    symmetric_reconstructed = dequantize_symmetric(
        symmetric_codes,
        symmetric_scale,
    )

    original = values.to(torch.float64)

    print(f"\n{'=' * 55}")
    print(f"Bit width: {bits}")
    print(f"Input: {values.tolist()}")

    for name, reconstructed, codes in [
        ("Affine", affine_reconstructed, affine_codes),
        ("Symmetric", symmetric_reconstructed, symmetric_codes),
    ]:
        error = original - reconstructed

        mae = error.abs().mean().item()
        rmse = error.square().mean().sqrt().item()
        max_error = error.abs().max().item()

        print(f"\n{name}")
        print(f"  Codes:        {codes.tolist()}")
        print(f"  MAE:          {mae:.6f}")
        print(f"  RMSE:         {rmse:.6f}")
        print(f"  Max abs error:{max_error: .6f}")


# Controlled examples: identical input values for both methods.
test_tensors = {
    "balanced": torch.tensor(
        [-1.0, -0.5, 0.0, 0.5, 1.0]
    ),
    "asymmetric": torch.tensor(
        [-0.2, 0.1, 0.3, 0.4, 1.0]
    ),
    "outlier": torch.tensor(
        [-0.1, 0.1, 0.2, 0.3, 5.0]
    ),
}

for name, values in test_tensors.items():
    print(f"\n\nDATASET: {name}")

    compare_methods(values, bits=4, symmetric_qmax=7)
    compare_methods(values, bits=2, symmetric_qmax=1)

print("\nComparison complete.")
