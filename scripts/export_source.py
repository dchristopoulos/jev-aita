"""Export the pinned D-Lab Parquet file to the JSONL that jevbench.sample_2025 reads.

Needs `pip install pyarrow`. The output must hash to `raw_sha256` in
data/final-ucb-2025.source.json; the script exits non-zero if it does not.

    curl -L -o data/ucb-2025-source.parquet https://huggingface.co/datasets/ucberkeley-dlab/fragility-moral-judgment-llms/resolve/cb4c298cbfa93ce9cdf56685f12a55b0a6928110/dilemmas/train.parquet
    python scripts/export_source.py data/ucb-2025-source.parquet data/ucb-2025-raw.jsonl
"""

import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

src, out = map(Path, sys.argv[1:3])
rows = pq.read_table(src).to_pylist()
data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
out.write_bytes(data)
got = hashlib.sha256(data).hexdigest()
want = json.loads(Path("data/final-ucb-2025.source.json").read_text())["raw_sha256"]
print(f"{len(rows)} rows -> {out}  sha256 {got}")
sys.exit(0 if got == want else f"expected {want}")
