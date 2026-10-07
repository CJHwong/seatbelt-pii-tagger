# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers", "sentencepiece", "protobuf", "numpy", "pyarrow", "pyyaml"]
# ///
"""Turn the checked developer-work rows into a training split.

usage: uv run --script prepare.py   -> train.jsonl, prepare_stats.json

Reads reviewed/*.reviewed.jsonl. Rows with no mark are kept up to NEGATIVE_SHARE of each
writer's rows, picked by a hash of the id: unmarked rows teach the tagger to stay quiet,
which is the failure these rows target. The gates are the casual-text source's (see
data/synthetic/llm_social/prepare.py): a repeated text, all but the first row of a
near-duplicate cluster, and a row that shares a stretch of text with any evaluation set
are dropped. Every kept row trains; there is no calibration split.
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
SOCIAL = HERE.parent / "llm_social"
sys.path.insert(0, str(SOCIAL))

from prepare import NEAR_DUPLICATE, SPACE, clusters, evaluation_texts, parse_marked, windows

NEGATIVE_SHARE = 0.15


def capped(rows: list[dict], share: float = NEGATIVE_SHARE) -> list[dict]:
    """All rows with a span, and rows without one up to `share` of the result."""
    positive = [row for row in rows if any(s["label"] != "ignore" for s in row["spans"])]
    negative = sorted(
        (row for row in rows if row not in positive), key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest()
    )
    room = int(len(positive) * share / (1 - share))
    kept = {row["id"] for row in positive + negative[:room]}
    return [row for row in rows if row["id"] in kept]


def main() -> None:
    build(HERE, NEGATIVE_SHARE)


def build(here: Path, share: float, extra_evaluation: list[str] | None = None) -> None:
    """The gates above on here/reviewed, written to here/train.jsonl and prepare_stats.json.
    extra_evaluation adds texts to the evaluation-overlap gate."""
    rows, counts = [], Counter()
    for path in sorted((here / "reviewed").glob("*.reviewed.jsonl")):
        written = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                raw = json.loads(line)
                text, spans = parse_marked(raw["text"])
                written.append(
                    {
                        "id": raw["id"],
                        "register": raw["register"],
                        "script": raw["script"],
                        "text": text,
                        "spans": spans,
                        "marked": raw["text"],
                    }
                )
        kept = capped(written, share)
        counts["read"] += len(written)
        counts["negative_over_cap"] += len(written) - len(kept)
        rows += kept
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
    hit = {
        owner[window]
        for text in evaluation_texts() + (extra_evaluation or [])
        for window in windows(text)
        if window in owner
    }
    counts["evaluation_overlap"] = len(hit)
    kept = [row for index, row in enumerate(unique) if index not in hit]
    for row in kept:
        row.pop("marked")
    (here / "train.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept), encoding="utf-8"
    )
    counts["train"] = len(kept)
    counts.update(f"train_{row['script']}" for row in kept)
    (here / "prepare_stats.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
