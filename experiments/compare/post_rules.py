# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers", "numpy", "pyarrow", "pyyaml"]
# ///
"""Apply the guideline's form rules to saved predictions, and write them as a new system.

usage: uv run --script post_rules.py apply SYSTEM   -> runs/[GROUP/]SYSTEM_post.jsonl
       uv run --script post_rules.py tool SYSTEM    -> runs/[GROUP/]SYSTEM_tool.jsonl (tool-output shapes)
       uv run --script post_rules.py keys SYSTEM    -> runs/[GROUP/]SYSTEM_keys.jsonl (handles in username fields)
       uv run --script post_rules.py dev            -> gold spans of the dev split the rules drop

`prune_lib.post_rules` drops a place name with no street part, a public-host URL, a shared
or reserved email, a default account, and a username inside a dropped URL. dev measures
the recall cost on the training sources before any evaluation set is scored.
"""

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path[:0] = [str(ROOT / "experiments/first_run"), str(ROOT / "experiments/vocab_prune"), str(HERE)]

from prune_lib import post_rules, tool_rules, username_keys

GROUPS = ["comparison", "generalization", "blind", "rules", "benchmarks", "published"]
RULED_LABELS = ("address", "url", "email", "username")


def run_dir(group: str) -> Path:
    return HERE / "runs" if group == "comparison" else HERE / "runs" / group


def apply(system: str, rules=post_rules, suffix: str = "_post") -> None:
    from sets import GROUPS as SET_GROUPS

    for group in GROUPS:
        sets = SET_GROUPS[group]()
        records = [json.loads(line) for line in (run_dir(group) / f"{system}.jsonl").open(encoding="utf-8")]
        dropped: Counter = Counter()
        added: Counter = Counter()
        with open(run_dir(group) / f"{system}{suffix}.jsonl", "w", encoding="utf-8") as handle:
            for record in records:
                text = sets[record["set"]][record["index"]]["text"]
                kept = rules(record["spans"], text)
                dropped.update(span["label"] for span in record["spans"] if span not in kept)
                added.update(span["label"] for span in kept if span not in record["spans"])
                handle.write(json.dumps({**record, "spans": kept}) + "\n")
        print(group, len(records), "rows, dropped", dict(dropped), "added", dict(added))


def dev() -> None:
    """Treat the gold spans as predictions: every gold span the rules drop is a lost hit."""
    gold: Counter = Counter()
    lost: Counter = Counter()
    examples: dict[str, list[str]] = {label: [] for label in RULED_LABELS}
    for line in (ROOT / "data/prepared/dev.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        spans = [entity for entity in row["entities"] if entity["label"] in RULED_LABELS]
        kept = post_rules(spans, row["text"])
        gold.update(span["label"] for span in spans)
        for span in spans:
            if span in kept:
                continue
            lost[span["label"]] += 1
            examples[span["label"]].append(row["text"][span["start"] : span["end"]])
    for label in RULED_LABELS:
        share = lost[label] / gold[label] if gold[label] else 0.0
        print(f"{label}: {lost[label]} of {gold[label]} gold spans dropped ({share:.2%})")
    (HERE / "runs" / "post_rules_dev_losses.json").write_text(json.dumps(examples, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "dev":
        dev()
    elif sys.argv[1] == "tool":
        apply(sys.argv[2], tool_rules, "_tool")
    elif sys.argv[1] == "keys":
        apply(sys.argv[2], username_keys, "_keys")
    else:
        apply(sys.argv[2])
