"""Print schema and one example of the raw parquet (no image decoding)."""
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
f = ROOT / "data/raw/RICO-Screen2Words/data/val-00000-of-00002.parquet"
pf = pq.ParquetFile(f)
print(pf.schema_arrow)
cols = [c for c in pf.schema_arrow.names if not c.startswith("image")]
t = pf.read_row_group(0, columns=cols).slice(0, 2).to_pylist()
r = t[0]
for k, v in r.items():
    s = v if not isinstance(v, str) else v[:300]
    print(f"== {k}: {s}")
sa = json.loads(r["semantic_annotations"])
print("SA keys:", list(sa.keys()))
def walk(n, d=0, out=[]):
    out.append((d, {k: v for k, v in n.items() if k != "children"}))
    for c in n.get("children", []) or []:
        walk(c, d + 1, out)
    return out
for d, n in walk(sa)[:12]:
    print("  " * d, n)
vh = json.loads(r["view_hierarchy"])
print("VH top keys:", list(vh.keys()))
root = vh.get("activity", {}).get("root", vh)
print("VH root keys:", list(root.keys()))
