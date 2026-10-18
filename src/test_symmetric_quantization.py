
import torch

from symmetric_quantization import (
    quantize_symmetric,
    dequantize_symmetric,
)


def test_symmetric(
    values: torch.Tensor,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    q, scale = quantize_symmetric(values, qmin, qmax)
    reconstructed = dequantize_symmetric(q, scale)

    print("\nOriginal:      ", values.tolist())
    print("Integer range: ", (qmin, qmax))
    print("Scale:         ", scale.item())
    print("Zero point:    ", 0)
    print("Codes:         ", q.tolist())
    print("Reconstructed: ", reconstructed.tolist())
    print(
        "Absolute error:",
        (values.double() - reconstructed).abs().tolist(),
    )

    assert q.min().item() >= qmin
    assert q.max().item() <= qmax
    assert torch.isfinite(scale).all()
    assert scale.item() > 0
    assert torch.isfinite(reconstructed).all()

    return reconstructed


# Test 1: Ordinary values.
print("TEST 1: Symmetric quantization")

values = torch.tensor(
    [-1.0, -0.5, 0.0, 0.5, 1.0],
    dtype=torch.float32,
)

reconstructed = test_symmetric(values, -2, 2)

assert torch.allclose(
    reconstructed,
    values.double(),
    atol=1e-6,
    rtol=1e-6,
)

print("\nSymmetric reconstruction test passed.")


# Test 2: An asymmetric distribution.
# This lets us observe how symmetric quantization handles
# values that do not have balanced positive and negative ranges.
print("\nTEST 2: Asymmetric data distribution")

values = torch.tensor(
    [-0.2, 0.1, 0.3, 0.4, 1.0],
    dtype=torch.float32,
)

test_symmetric(values, -7, 7)


# Test 3: All-zero tensor.
print("\nTEST 3: All-zero tensor")

values = torch.zeros(4, dtype=torch.float32)
reconstructed = test_symmetric(values, -7, 7)

assert torch.equal(reconstructed, values.double())

print("\nAll symmetric quantization tests passed.")
