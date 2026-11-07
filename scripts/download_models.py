"""Download Gemma 4 QAT (q4_0-unquantized, bf16 weights trained for 4-bit) checkpoints into models/.
Only weights/config/tokenizer files are fetched; nothing is executed (no trust_remote_code).
Usage: .venv/Scripts/python.exe -I scripts/download_models.py E2B [E4B 12B]
"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
from huggingface_hub import snapshot_download
REPOS = {"E2B": "google/gemma-4-E2B-it-qat-q4_0-unquantized",
         "E4B": "google/gemma-4-E4B-it-qat-q4_0-unquantized",
         "12B": "google/gemma-4-12B-it-qat-q4_0-unquantized"}
for k in sys.argv[1:]:
    p = snapshot_download(REPOS[k], local_dir=ROOT / "models" / REPOS[k].split("/")[1],
                          allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.model", "README.md"])
    print(k, p, flush=True)
