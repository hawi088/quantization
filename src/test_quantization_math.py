
import torch

from quantization_math import (
    quantize_affine,
    dequantize_affine,
)


def test_tensor(values: torch.Tensor, qmin: int, qmax: int) -> None:
    q, scale, zero_point = quantize_affine(values, qmin, qmax)
    reconstructed = dequantize_affine(q, scale, zero_point)

    print("\nOriginal values: ", values.tolist())
    print("Integer range:   ", (qmin, qmax))
    print("Scale:           ", scale.item())
    print("Zero point:      ", zero_point)
    print("Quantized codes: ", q.tolist())
    print("Reconstructed:   ", reconstructed.tolist())
    print(
        "Absolute errors: ",
        (values.double() - reconstructed).abs().tolist(),
    )

    # Codes must stay within the requested integer range.
    assert q.min().item() >= qmin
    assert q.max().item() <= qmax

    # Scale must be finite and positive.
    assert torch.isfinite(scale).all()
    assert scale.item() > 0

    # Reconstruction must be finite.
    assert torch.isfinite(reconstructed).all()

    return reconstructed


# Test 1: Ordinary, nonconstant tensor.
print("TEST 1: Ordinary affine quantization")

values = torch.tensor(
    [-1.0, -0.5, 0.0, 0.5, 1.0],
    dtype=torch.float32,
)

reconstructed = test_tensor(values, qmin=-2, qmax=1)

print("\nBasic mathematical checks passed.")


# Test 2: Constant positive, negative, and zero tensors.
print("\nTEST 2: Constant tensors")

for constant in [3.5, -2.5, 0.0]:
    values = torch.full((4,), constant, dtype=torch.float32)

    reconstructed = test_tensor(values, qmin=-2, qmax=1)

    assert torch.allclose(
        reconstructed,
        values.double(),
        atol=1e-6,
        rtol=1e-6,
    ), f"Constant tensor {constant} was not reconstructed correctly."

    print(f"Constant {constant} reconstructed correctly.")

print("\nAll quantization mathematics tests passed.")
