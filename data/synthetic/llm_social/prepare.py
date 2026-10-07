# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers", "sentencepiece", "protobuf", "numpy", "pyarrow", "pyyaml"]
# ///
"""Turn the checked LLM-written rows into a training split and a calibration split.

usage: uv run --script prepare.py [--exclude WRITER]...   -> train.jsonl, calibration.jsonl, prepare_stats.json

Reads reviewed/*.reviewed.jsonl, except the writers named with --exclude, which isolates
one writer's effect in an A/B run. Drops a row that repeats another row's text, all but the
first row of a near-duplicate cluster (near_duplicates.py at NEAR_DUPLICATE), and a row
that shares a stretch of text with any evaluation set, after spaces are removed: HAN_WINDOW
characters when the stretch has a Han character, LATIN_WINDOW otherwise. 16 Latin
characters are only three or four words, and common phrases matched at that length. A hash of the row id puts CALIBRATION_SHARE of the rows in calibration.jsonl,
which never trains: compare/calibrate.py fits its maps there, on text closer to social
deployment than the dev split.
"""

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path[:0] = [
    str(HERE),
    str(ROOT / "benchmark"),
    str(ROOT / "experiments/compare"),
    str(ROOT / "experiments/vocab_prune"),
]

from build import parse_marked
from data import zh_bench_rows
from near_duplicates import clusters
from sets import GROUPS

HAN_WINDOW = 16
LATIN_WINDOW = 32
HAN = re.compile(r"[\u4e00-\u9fff]")
CALIBRATION_SHARE = 0.2
NEAR_DUPLICATE = 0.5
SPACE = re.compile(r"\s+")


def windows(text: str) -> set[str]:
    flat = SPACE.sub("", text)
    han = {flat[i : i + HAN_WINDOW] for i in range(len(flat) - HAN_WINDOW + 1)}
    latin = {flat[i : i + LATIN_WINDOW] for i in range(len(flat) - LATIN_WINDOW + 1)}
    return {window for window in han if HAN.search(window)} | {window for window in latin if not HAN.search(window)}


def evaluation_texts() -> list[str]:
    texts = [row["text"] for loader in GROUPS.values() for rows in loader().values() for row in rows]
    return texts + [row["text"] for row in zh_bench_rows()]


def in_calibration(row_id: str) -> bool:
    return int(hashlib.sha256(row_id.encode()).hexdigest(), 16) % 1000 < CALIBRATION_SHARE * 1000


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude", action="append", default=[])
    excluded = set(parser.parse_args().exclude)
    rows = []
    for path in sorted((HERE / "reviewed").glob("*.reviewed.jsonl")):
        if path.name.removesuffix(".reviewed.jsonl") in excluded:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                raw = json.loads(line)
                text, spans = parse_marked(raw["text"])
                rows.append(
                    {
                        "id": raw["id"],
                        "register": raw["register"],
                        "script": raw["script"],
                        "text": text,
                        "spans": spans,
                        "marked": raw["text"],
                    }
                )
    counts: Counter = Counter(read=len(rows))
    seen, unique = set(), []
    for row in rows:
        key = SPACE.sub("", row["text"])
        if key in seen:
            counts["duplicate"] += 1
            continue
        seen.add(key)
        unique.append(row)
    marked = [{"script": row["script"], "text": row["marked"]} for row in unique]
    later = {index for group in clusters(marked, NEAR_DUPLICATE) for index in sorted(group)[1:]}
    counts["near_duplicate"] = len(later)
    unique = [row for index, row in enumerate(unique) if index not in later]
    owner: dict[str, int] = {}
    for index, row in enumerate(unique):
        for window in windows(row["text"]):
            owner.setdefault(window, index)
    hit = set()
    for text in evaluation_texts():
        for window in windows(text):
            if window in owner:
                hit.add(owner[window])
    counts["evaluation_overlap"] = len(hit)
    kept = [row for index, row in enumerate(unique) if index not in hit]
    splits = {"train": [], "calibration": []}
    for row in kept:
        row.pop("marked")
        splits["calibration" if in_calibration(row["id"]) else "train"].append(row)
    for name, split in splits.items():
        body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in split)
        (HERE / f"{name}.jsonl").write_text(body, encoding="utf-8")
        counts[name] = len(split)
        counts.update(f"{name}_{row['script']}" for row in split)
    (HERE / "prepare_stats.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts))
    for index in sorted(hit)[:20]:
        print("overlap:", unique[index]["id"], unique[index]["text"][:60].replace("\n", " "))


if __name__ == "__main__":
    main()
