#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=0.34", "pyyaml>=6", "pyarrow>=17"]
# ///
"""Check what fetch_data.py put on disk, and verify the eval sets are usable.

Presence and size for every dataset. Then, for datasets that carry character
offsets, every span is validated by slicing the source text: a benchmark whose
offsets do not line up is worse than no benchmark, because it silently mis-scores
every model you point at it.

Usage:
    uv run scripts/verify_data.py
    uv run scripts/verify_data.py --deep      # validate every span in every set
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "datasets.yaml"
INVENTORY = ROOT / "data" / "INVENTORY.json"

# Where the text and the spans live inside each dataset, so the offset check is
# generic rather than one branch per dataset.
SHAPES = {
    "lianghsun/tw-PII-bench": {"text": "text", "spans": "spans", "start": "start", "end": "end", "label": "label"},
    "wan9yu/pii-bench-zh": {"text": "text", "spans": "entities", "start": "start", "end": "end", "label": "type"},
}


def load_manifest() -> tuple[Path, list[dict]]:
    with MANIFEST.open() as handle:
        manifest = yaml.safe_load(handle)
    root = ROOT / (os.environ.get("PII_MODEL_DATA_ROOT") or manifest["data_root_default"])  # an absolute value wins
    return root, manifest["datasets"]


def iter_rows(path: Path):
    if path.suffix == ".parquet":
        yield from pq.read_table(path).to_pylist()
        return
    if path.suffix == ".jsonl":
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        yield from (data if isinstance(data, list) else [data])


def check_offsets(dataset_dir: Path, shape: dict, deep: bool) -> tuple[int, int]:
    good = bad = 0
    for path in sorted(dataset_dir.rglob("*")):
        if path.suffix not in {".parquet", ".jsonl", ".json"}:
            continue
        for row in iter_rows(path):
            text = row.get(shape["text"])
            if not isinstance(text, str):
                continue
            for span in row.get(shape["spans"]) or []:
                if shape["start"] not in span:
                    continue
                sliced = text[span[shape["start"]] : span[shape["end"]]]
                if sliced == span.get("text"):
                    good += 1
                else:
                    bad += 1
                    if bad <= 3:
                        print(
                            f"      MISMATCH {path.name}: "
                            f"{span.get(shape['label'])!r} offset "
                            f"[{span[shape['start']]}:{span[shape['end']]}] "
                            f"expected {span.get('text')!r} got {sliced!r}"
                        )
            if not deep and good > 5000:
                return good, bad
    return good, bad


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep", action="store_true", help="validate every span, not the first few thousand")
    args = parser.parse_args()

    root, datasets = load_manifest()
    inventory = json.loads(INVENTORY.read_text()) if INVENTORY.exists() else {}

    print(f"data root: {root}\n")
    print(f"{'dataset':52s} {'role':6s} {'rev':9s} {'size':>10s}  offsets")
    print("-" * 100)

    problems = []
    gated = []
    for spec in datasets:
        repo_id = spec["id"]
        target = root / repo_id
        record = inventory.get(repo_id, {})
        size = record.get("bytes", 0)

        # A gated dataset that is absent is an expected gap, not a failure. Reporting
        # it as a failure would train the reader to ignore this script's exit code.
        if spec.get("gated") and size == 0:
            gated.append(repo_id)
            print(f"{repo_id:52s} {spec.get('role', ''):6s} {'-':9s} {'GATED':>10s}  -")
            continue

        if not target.exists() or size == 0:
            problems.append(f"{repo_id}: not downloaded")
            print(f"{repo_id:52s} {spec.get('role', ''):6s} {'-':9s} {'MISSING':>10s}  -")
            continue

        if repo_id in SHAPES:
            good, bad = check_offsets(target, SHAPES[repo_id], args.deep)
            offsets = f"{good} ok, {bad} BAD" if good or bad else "no spans found"
            if bad:
                problems.append(f"{repo_id}: {bad} bad offsets")
            if not good:
                problems.append(f"{repo_id}: no spans found, the schema may have changed")
        else:
            offsets = "-"

        print(
            f"{repo_id:52s} {spec.get('role', ''):6s} "
            f"{(record.get('revision') or '')[:8]:9s} {size / 1e6:9.1f}M  {offsets}"
        )

        declared = spec.get("rows")
        if declared and repo_id == "lianghsun/tw-PII-bench":
            actual = sum(pq.read_table(p).num_rows for p in target.rglob("*.parquet"))
            if actual != declared:
                problems.append(f"{repo_id}: manifest says {declared} rows, disk has {actual}")

    total = sum(r.get("bytes", 0) for r in inventory.values())
    print(f"\ntotal on disk: {total / 1e9:.2f} GB")
    if gated:
        print("\ngated, needs manual approval on Hugging Face before it will download:")
        for item in gated:
            print(f"  {item}")
        print("After approval, re-run: uv run scripts/fetch_data.py --only <id>")
    if problems:
        print("\nproblems:")
        for item in problems:
            print(f"  {item}")
        return 1
    print("\nevery non-gated dataset is present, and all checked offsets slice correctly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
