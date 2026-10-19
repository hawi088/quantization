
import torch

from quantizers import quantize_tensor, dequantize_tensor


def test_tensor(name, values, bits=8):
    original = torch.tensor(values, dtype=torch.float32)

    result = quantize_tensor(
        original,
        bits=bits,
        method="affine",
    )
    reconstructed = dequantize_tensor(result)

    print(f"\n{name}")
    print(f"Original:      {original.tolist()}")
    print(f"Quantized:     {result.codes.tolist()}")
    print(f"Range:         [{result.qmin}, {result.qmax}]")
    print(f"Scale:         {result.scale.item():.8f}")
    print(f"Zero point:    {result.zero_point}")
    print(f"Reconstructed: {reconstructed.tolist()}")

    assert result.codes.min().item() >= result.qmin
    assert result.codes.max().item() <= result.qmax
    assert torch.isfinite(reconstructed).all()
    assert result.scale.item() > 0

    print("PASS: codes and reconstruction are valid.")


def main():
    test_tensor("Mixed positive and negative", [-1.0, -0.2, 0.0, 0.4, 1.0])
    test_tensor("Positive-only", [0.1, 0.2, 0.5, 1.0])
    test_tensor("Negative-only", [-1.0, -0.5, -0.2, -0.1])
    test_tensor("Uneven range", [-3.0, -0.1, 0.0, 0.2, 7.0])


if __name__ == "__main__":
    main()
