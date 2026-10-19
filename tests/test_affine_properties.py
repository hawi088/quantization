
import torch

from quantizers import quantize_tensor, dequantize_tensor


def test_tensor(name, values):
    original = torch.tensor(values, dtype=torch.float32)

    result = quantize_tensor(original, bits=8, method="affine")
    reconstructed = dequantize_tensor(result)

    error = (original.double() - reconstructed).abs()
    max_error = error.max().item()

    print(f"\n{name}")
    print(f"Original:      {original.tolist()}")
    print(f"Quantized:     {result.codes.tolist()}")
    print(f"Scale:         {result.scale.item():.8f}")
    print(f"Zero point:    {result.zero_point}")
    print(f"Reconstructed: {reconstructed.tolist()}")
    print(f"Maximum error: {max_error:.8f}")

    assert result.codes.min().item() >= result.qmin
    assert result.codes.max().item() <= result.qmax
    assert torch.isfinite(reconstructed).all()
    assert result.scale.item() > 0

    # Every value should reconstruct within half a quantization
    # step, allowing a tiny floating-point tolerance.
    assert max_error <= result.scale.item() / 2 + 1e-7

    print("PASS")


def main():
    test_tensor("Mixed signs", [-1.0, -0.2, 0.0, 0.4, 1.0])
    test_tensor("Positive only", [0.1, 0.2, 0.5, 1.0])
    test_tensor("Negative only", [-1.0, -0.5, -0.2, -0.1])
    test_tensor("Uneven range", [-3.0, -0.1, 0.0, 0.2, 7.0])
    test_tensor("Positive constant", [0.5, 0.5, 0.5, 0.5])
    test_tensor("Negative constant", [-0.5, -0.5, -0.5, -0.5])
    test_tensor("Zero constant", [0.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    main()
