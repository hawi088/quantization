
import torch

from representations import get_integer_range, convert_float


print("INTEGER CODE RANGES")

for bits in (2, 4, 8):
    affine_range = get_integer_range(bits)
    symmetric_range = get_integer_range(bits, symmetric=True)

    print(
        f"INT{bits}: "
        f"affine={affine_range}, "
        f"symmetric={symmetric_range}"
    )

print("\nFLOATING-POINT CONVERSIONS")

values = torch.tensor(
    [1.0, -1.0, 0.3333, 100.25],
    dtype=torch.float32,
)

for precision in ("fp16", "bf16"):
    converted = convert_float(values, precision)

    print(f"\n{precision.upper()}")
    print("Dtype:      ", converted.dtype)
    print("Values:     ", converted.tolist())
    print(
        "Abs errors: ",
        (values - converted.float()).abs().tolist(),
    )

    assert converted.dtype in (torch.float16, torch.bfloat16)

print("\nAll representation tests passed.")
