# /// script
# requires-python = ">=3.12"
# dependencies = ["pyarrow", "pyyaml"]
# ///
"""Build the 12-type training set from the source datasets, as mapped in type_map.yaml.

usage: uv run scripts/build_training_set.py
Writes data/prepared/train.jsonl, data/prepared/dev.jsonl and data/prepared/stats.json, then
runs experiments/compare/leak_check.py, which rewrites runs/leaked.json and fails the build
when a training row copies a transcript dev, test, live or events window.
Each row: {"text", "entities": [{start, end, label}], "ignore": [{start, end}], "source", "script"}.
"""

import ast
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

import yaml
from pyarrow import parquet

sys.path.insert(0, str(Path(__file__).parent))
from prep_lib import (
    SPECIAL_KINDS,
    TYPES,
    find_all,
    is_dev,
    join_parts,
    map_spans,
    script_of,
    settle_overlaps,
    spans_valid,
    validate_map,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "prepared"
LOCAL_MAP = ROOT / "type_map.local.yaml"
KIND_PRIORITY = {kind: rank for rank, kind in enumerate([*TYPES, *SPECIAL_KINDS])}


def read_openpii() -> Iterator[dict]:
    with open(DATA / "ai4privacy/pii-masking-openpii-1.5m/data/train.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            yield {"text": row["source_text"], "spans": row["privacy_mask"], "language": row.get("language")}


def read_wismut() -> Iterator[dict]:
    with open(DATA / "Wismut/nym-pii-multilingual-data/train.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            yield {"text": row["text"], "spans": row["entities"], "language": None}


def read_nemotron() -> Iterator[dict]:
    for path in sorted((DATA / "nvidia/Nemotron-PII/data").glob("train-*.parquet")):
        table = parquet.read_table(path, columns=["text", "spans"])
        for text, spans in zip(table.column("text").to_pylist(), table.column("spans").to_pylist(), strict=True):
            yield {"text": text, "spans": ast.literal_eval(spans) if spans else [], "language": None}


def read_gretel_finance() -> Iterator[dict]:
    base = DATA / "gretelai/synthetic_pii_finance_multilingual/data"
    for path in sorted(base.glob("*_train-*.parquet")):
        if path.name.startswith("combined_"):
            continue
        table = parquet.read_table(path, columns=["generated_text", "pii_spans", "language"])
        columns = zip(
            table.column("generated_text").to_pylist(),
            table.column("pii_spans").to_pylist(),
            table.column("language").to_pylist(),
            strict=True,
        )
        for text, spans, language in columns:
            yield {"text": text, "spans": json.loads(spans) if spans else [], "language": language}


def read_gretel_en() -> Iterator[dict]:
    for path in sorted((DATA / "gretelai/gretel-pii-masking-en-v1/data").glob("train-*.parquet")):
        table = parquet.read_table(path, columns=["text", "entities"])
        for text, entities in zip(table.column("text").to_pylist(), table.column("entities").to_pylist(), strict=True):
            yield {"text": text, "entities_by_string": ast.literal_eval(entities) if entities else [], "language": None}


def read_synthetic() -> Iterator[dict]:
    """Only the train split. heldout is the clean Traditional test set and never trains."""
    with open(DATA / "synthetic/train.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            yield {"text": row["text"], "spans": row["spans"], "language": None, "script": "zh-Hant"}


def read_rule_secrets() -> Iterator[dict]:
    """Only the train split. heldout holds providers held out of training and never trains."""
    with open(DATA / "synthetic/rule_secrets/train.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            yield {
                "text": row["text"],
                "spans": row["spans"],
                "language": None,
                "script": row.get("script"),
                "kind": "tool_output",
            }  # code and config text


def read_llm_social() -> Iterator[dict]:
    """Social-media rows LLM agents wrote and a second model checked. Only the train split:
    calibration.jsonl is held out for compare/calibrate.py and never trains."""
    with open(DATA / "synthetic/llm_social/train.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            yield {"text": row["text"], "spans": row["spans"], "language": None, "script": row["script"]}


AGENT_PROSE = {
    "agent reply",
    "user prompt",
    "status report",
    "status rollup",
    "sprint board",
    "release checklist",
    "incident tracker",
    "vendor comparison",
    "on-call rotation",
    "review queue",
}


def read_llm_agentwork() -> Iterator[dict]:
    """Coding-agent sessions LLM agents wrote and a second model checked. A register that is not
    prose (a tool call, CLI output, JSON, a log) reads as a tool window to a run trained with --frame."""
    path = DATA / "synthetic/llm_agentwork/train.jsonl"
    if not path.exists():
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            kind = None if row.get("register") in AGENT_PROSE else "tool_output"
            yield {"text": row["text"], "spans": row["spans"], "language": None, "script": row["script"], "kind": kind}


def read_llm_devwork() -> Iterator[dict]:
    """Developer work prose LLM agents wrote and a second model checked."""
    path = DATA / "synthetic/llm_devwork/train.jsonl"
    if not path.exists():
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            yield {"text": row["text"], "spans": row["spans"], "language": None, "script": row["script"]}


TOOL_KINDS = {"tool_call", "tool_output"}


def read_path(path: Path) -> Iterator[dict]:
    """Rows already in training form, from a type_map `path:` dataset. `origin`, when a row
    has it, names the window a copy came from, so the copy follows it into the same split."""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            yield {
                "text": row["text"],
                "spans": row["spans"],
                "language": None,
                "script": row["script"],
                "origin": row.get("origin"),
                "kind": row.get("kind"),
            }


def read_wiki_negatives() -> Iterator[dict]:
    """Wikipedia paragraphs a checkpoint flagged. Public names and places train as not PII."""
    with open(DATA / "wikimedia/wiki_negatives/train.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            yield {"text": row["text"], "spans": [], "language": None, "script": row["script"]}


def read_wiki_negatives_no_person() -> Iterator[dict]:
    """The same paragraphs without the ones flagged as person. Public and private names look alike."""
    with open(DATA / "wikimedia/wiki_negatives/train.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if "person" not in row["flagged"]:
                yield {"text": row["text"], "spans": [], "language": None, "script": row["script"]}


READERS = {
    "ai4privacy/pii-masking-openpii-1.5m": read_openpii,
    "Wismut/nym-pii-multilingual-data": read_wismut,
    "nvidia/Nemotron-PII": read_nemotron,
    "gretelai/synthetic_pii_finance_multilingual": read_gretel_finance,
    "gretelai/gretel-pii-masking-en-v1": read_gretel_en,
    "synthetic/tw-templates": read_synthetic,
    "synthetic/rule-secrets": read_rule_secrets,
    "synthetic/llm-social": read_llm_social,
    "synthetic/llm-devwork": read_llm_devwork,
    "synthetic/llm-agentwork": read_llm_agentwork,
    "wikimedia/wiki-negatives": read_wiki_negatives,
    "wikimedia/wiki-negatives-no-person": read_wiki_negatives_no_person,
}


def spans_from_strings(text: str, entities: list[dict], label_map: dict[str, str]) -> list[dict] | None:
    """Locate string-only entities. None when an entity string does not occur in the text."""
    spans = []
    for entity in entities:
        label = min(entity["types"], key=lambda name: KIND_PRIORITY[label_map[name]])
        found = find_all(text, entity["entity"])
        if not found:
            return None
        spans += [{"start": start, "end": end, "label": label} for start, end in found]
    return spans


def offsets_match_values(text: str, spans: list[dict]) -> bool:
    """Where a source records the entity text, the offsets must point at exactly that text."""
    return all(text[s["start"] : s["end"]] == s["value"] for s in spans if "value" in s)


def keep_row(script: str | None, language: str | None, config: dict) -> str | None:
    """None when the row is kept, else the reason it is dropped."""
    if script is None:
        return "script_outside_target"
    if script.startswith("zh"):
        return None if config["keep_chinese"] else "chinese_not_wanted"
    if config["languages"] and language not in config["languages"]:
        return "language_filtered"
    return None


def process(row: dict, config: dict) -> tuple[dict | None, str | None, int]:
    """One source row to one training row. Returns (row or None, drop reason, overlaps dropped)."""
    text = row["text"] or ""
    script = row.get("script") or script_of(text)  # a source may know its script better than the markers
    reason = keep_row(script, row["language"], config)
    if reason:
        return None, reason, 0
    label_map = config["labels"]
    if "entities_by_string" in row:
        raw = spans_from_strings(text, row["entities_by_string"], label_map)
        if raw is None:
            return None, "entity_string_not_found", 0
    else:
        raw = row["spans"]
    if not spans_valid(text, raw) or not offsets_match_values(text, raw):
        return None, "offsets_invalid", 0
    entities, ignored, overlaps = settle_overlaps(join_parts(text, map_spans(raw, label_map)))
    return {"text": text, "entities": entities, "ignore": ignored, "script": script}, None, overlaps


def build(config: dict) -> dict:
    OUT.mkdir(exist_ok=True)
    stats: dict = {"datasets": {}, "excluded_datasets": {}}
    seen: set[bytes] = set()
    with (
        open(OUT / "train.jsonl", "w", encoding="utf-8") as train,
        open(OUT / "dev.jsonl", "w", encoding="utf-8") as dev,
    ):
        for name, dataset in config["datasets"].items():
            if not dataset["include"]:
                stats["excluded_datasets"][name] = " ".join(dataset["reason"].split())
                continue
            if dataset.get("private") and not (ROOT / dataset["path"]).exists():
                print(
                    f"warning: skipping {name}: private file {dataset['path']} is not on this machine", file=sys.stderr
                )
                stats["excluded_datasets"][name] = "private file absent"
                continue
            validate_map(dataset["labels"])
            stats["datasets"][name] = build_one(name, dataset, seen, train, dev)
    return stats


def build_one(name: str, config: dict, seen: set[bytes], train, dev) -> dict:
    counts: Counter = Counter()
    entities_by_script: dict[str, Counter] = defaultdict(Counter)
    rows = read_path(ROOT / config["path"]) if "path" in config else READERS[name]()
    for row in rows:
        counts["read"] += 1
        record, reason, overlaps = process(row, config)
        counts["overlapping_entities_dropped"] += overlaps
        if record is None:
            counts[f"dropped_{reason}"] += 1
            continue
        digest = hashlib.sha256(record["text"].encode("utf-8")).digest()
        if digest in seen:
            counts["dropped_duplicate_text"] += 1
            continue
        seen.add(digest)
        record["source"] = name
        if row.get("kind") in TOOL_KINDS:  # read as the segment id by a run trained with --frame
            record["frame"] = 1
        split = dev if is_dev(row.get("origin") or record["text"]) else train  # a copy follows its first version
        split.write(json.dumps(record, ensure_ascii=False) + "\n")
        counts["kept_dev" if split is dev else "kept_train"] += 1
        counts["ignore_spans"] += len(record["ignore"])
        entities_by_script[record["script"]].update(entity["label"] for entity in record["entities"])
    return {"counts": dict(counts), "entities_by_script": {k: dict(v) for k, v in entities_by_script.items()}}


def main() -> None:
    config = yaml.safe_load((ROOT / "type_map.yaml").read_text())
    if LOCAL_MAP.exists():  # sources kept off the release; a local entry with the same name wins
        config["datasets"] |= yaml.safe_load(LOCAL_MAP.read_text())["datasets"]
    stats = build(config)
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    check = subprocess.run(["uv", "run", "--script", "leak_check.py"], cwd=ROOT / "experiments/compare", check=False)
    if check.returncode != 0:
        sys.exit(f"leak check failed with exit {check.returncode}; see its output above")


if __name__ == "__main__":
    main()
