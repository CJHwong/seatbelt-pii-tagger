# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers", "sentencepiece", "protobuf", "numpy", "pyarrow", "pyyaml", "huggingface_hub"]
# ///
"""Does any training or dev row share text with an evaluation text?

usage: uv run --script leak_check.py   -> runs/leaked.json, counts printed; exit 1 on a transcript leak

Evaluation texts: every group in sets.GROUPS, zh_bench, and the transcript dev and test
windows. A training row covers an evaluation text when it holds the same whitespace-free
text, or shares 20% or more of the text's windows of 32 Latin or 16 Han characters.

runs/leaked.json lists, per set, the text_key of every evaluation text a row copies or
covers by half or more. sets.py marks those rows and score.py leaves them out, so a set
that shares a generator with a training source scores only its unseen rows. The transcript
windows are never listed: data/dev_transcripts/prepare.py keeps them out of training, and a
copy or a half-covered window here exits 1. Prints counts only, never text.
"""

import json
import multiprocessing
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "data/synthetic/llm_social"), str(ROOT / "experiments/vocab_prune")]
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from compare_lib import text_key

PREPARED = ROOT / "data/prepared"
TRANSCRIPTS = HERE / "runs/transcripts"
OUT = HERE / "runs/leaked.json"
SPACE, HAN = re.compile(r"\s+"), re.compile(r"[一-鿿]")
WINDOW_SIZES = ((16, True), (32, False))  # (characters, Han window)
COVERED, LEAKED = 0.2, 0.5
TRANSCRIPT_SETS = ("transcript_dev", "transcript_test")

# The index, built once in the parent; forked workers share it.
# Every owner, not the first: a text in two sets must be marked leaked in both.
window_owner: dict[str, list[int]] = defaultdict(list)
flat_owner: dict[str, list[int]] = defaultdict(list)
set_of: list[str] = []
window_count: list[int] = []
key_of: list[str] = []


def windows(flat: str) -> set[str]:
    found = set()
    for size, han in WINDOW_SIZES:
        for start in range(len(flat) - size + 1):
            window = flat[start : start + size]
            if bool(HAN.search(window)) == han:
                found.add(window)
    return found


def add(text: str, name: str) -> None:
    flat = SPACE.sub("", text)
    index = len(set_of)
    flat_owner[flat].append(index)
    found = windows(flat)
    set_of.append(name)
    window_count.append(len(found))
    key_of.append(text_key(text))
    for window in found:
        window_owner[window].append(index)


def evaluation_texts() -> None:
    from data import zh_bench_rows
    from sets import GROUPS  # first: it puts the benchmark folders on the path

    for load in GROUPS.values():
        for name, rows in load().items():
            for row in rows:
                add(row["text"], name)
    for row in zh_bench_rows():
        add(row["text"], "zh_bench")
    for name, path in zip(TRANSCRIPT_SETS, ("sample.jsonl", "test/sample.jsonl"), strict=True):
        if not (TRANSCRIPTS / path).exists():  # private; a machine without it trains no transcript rows either
            print(f"warning: {name} is not on this machine; its leak check is skipped", file=sys.stderr)
            continue
        for line in (TRANSCRIPTS / path).open(encoding="utf-8"):
            add(json.loads(line)["text"], name)


def covered(line: str) -> list[tuple[str, int, str]]:
    """(training source, evaluation text, exact or half or more or under half) per text it covers."""
    row = json.loads(line)
    flat = SPACE.sub("", row["text"])
    if flat in flat_owner:
        return [(row["source"], index, "exact") for index in flat_owner[flat]]
    hits = Counter(index for w in windows(flat) if w in window_owner for index in window_owner[w])
    found = []
    for index, count in hits.items():
        share = count / max(window_count[index], 1)
        if share >= COVERED:
            found.append((row["source"], index, "half or more" if share >= LEAKED else "under half"))
    return found


def scan(split: str) -> list[tuple[str, int, str]]:
    context = multiprocessing.get_context("fork")  # macOS spawns by default, which rebuilds the index per worker
    with context.Pool(os.cpu_count()) as pool, (PREPARED / f"{split}.jsonl").open(encoding="utf-8") as handle:
        return [hit for hits in pool.imap(covered, handle, chunksize=2000) for hit in hits]


def report(split: str, hits: list[tuple[str, int, str]]) -> None:
    rows = Counter((source, set_of[index], kind) for source, index, kind in hits)
    texts = defaultdict(set)
    for source, index, kind in hits:
        texts[source, set_of[index], kind].add(index)
    print(f"{split}: {len(hits)} row and text pairs share {COVERED:.0%} or more")
    for (source, name, kind), count in sorted(rows.items(), key=lambda item: -item[1]):
        print(
            f"  {source} -> {name} ({kind}): {count} training rows, {len(texts[source, name, kind])} evaluation texts"
        )


def main() -> int:
    evaluation_texts()
    print(f"index: {len(set_of)} evaluation texts, {len(window_owner)} windows", flush=True)
    leaked: dict[str, set[str]] = defaultdict(set)
    for split in ("train", "dev"):
        hits = scan(split)
        report(split, hits)
        for _, index, kind in hits:
            if kind != "under half":
                leaked[set_of[index]].add(key_of[index])
    transcripts = {name: len(leaked.pop(name, ())) for name in TRANSCRIPT_SETS}
    OUT.write_text(json.dumps({name: sorted(keys) for name, keys in sorted(leaked.items())}, indent=1) + "\n")
    print("left out of scoring:", {name: len(keys) for name, keys in sorted(leaked.items())})
    if any(transcripts.values()):
        print(
            f"<leak_check_failed>transcript windows copied or half covered in training: {transcripts}</leak_check_failed>"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
