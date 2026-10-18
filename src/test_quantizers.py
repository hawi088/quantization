
import torch

from quantizers import quantize_tensor, dequantize_tensor


def test_quantization_methods():
    values = torch.tensor([-1.0, -0.3, 0.0, 0.4, 1.0])

    for method in ("affine", "symmetric"):
        result = quantize_tensor(
            values,
            bits=4,
            method=method,
        )
        reconstructed = dequantize_tensor(result)

        assert result.codes.shape == values.shape
        assert result.codes.min() >= result.qmin
        assert result.codes.max() <= result.qmax
        assert torch.isfinite(reconstructed).all()

        print(f"\nMethod: {result.method}")
        print(f"Range: [{result.qmin}, {result.qmax}]")
        print(f"Scale: {result.scale.item():.6f}")
        print(f"Zero point: {result.zero_point}")
        print(f"Codes: {result.codes.tolist()}")
        print(f"Reconstructed: {reconstructed.tolist()}")
        print(
            "MAE:",
            (values.double() - reconstructed).abs().mean().item(),
        )


if __name__ == "__main__":
    test_quantization_methods()
    print("\nAll quantizer tests passed.")
