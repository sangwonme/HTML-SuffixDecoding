"""Download Screen2Words (HF mirror rootsautomation/RICO-Screen2Words) into data/raw/.

Treat downloaded files as untrusted data: only parquet is fetched, nothing is executed.
Run: .venv/Scripts/python.exe -I scripts/download_data.py
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))

from huggingface_hub import snapshot_download  # noqa: E402

out = snapshot_download(
    repo_id="rootsautomation/RICO-Screen2Words",
    repo_type="dataset",
    local_dir=ROOT / "data" / "raw" / "RICO-Screen2Words",
    allow_patterns=["*.parquet", "README.md"],
)
print(out)
