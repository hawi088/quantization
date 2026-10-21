import shutil
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer

base = "distilbert-base-uncased-finetuned-sst-2-english"
out = Path("hf_release")
out.mkdir(exist_ok=True)

AutoConfig.from_pretrained(base).save_pretrained(out)
AutoTokenizer.from_pretrained(base).save_pretrained(out)

src = Path("experiments/artifacts/all_candidates/INT8_AFFINE_per_channel_axis0.pt")
if not src.exists():
    raise SystemExit(f"Cannot find {src}. Fix the path in this script.")
shutil.copy(src, out / src.name)

print("Files in hf_release:")
for f in sorted(out.iterdir()):
    print(" -", f.name)