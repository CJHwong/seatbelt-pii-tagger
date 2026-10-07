"""The agent-seatbelt rule fixtures as test sets, and the split of rule ids held out of training.

The rules engine ships one sample per provider-secret rule (271 rules from gitleaks,
betterleaks, Microsoft and seatbelt), and a false-positive suite labelled per row, not per
span. About one rule in five is held out of every training set built from the rules, so
recall on those samples tests providers the model never saw.
"""

import hashlib
import json
from pathlib import Path

from compare_lib import normalized

SEATBELT = Path(__file__).resolve().parents[2] / "data" / "external" / "agent-seatbelt" / "hooks-pii" / "tests"
SAMPLE_FILES = (
    "gitleaks-samples.jsonl",
    "betterleaks-samples.jsonl",
    "microsoft-samples.jsonl",
    "seatbelt-samples.jsonl",
)
HELD_OUT_SHARE = 0.2
GENERATED_HELD_OUT = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "rule_secrets" / "heldout.jsonl"


def held_out(rule_id: str) -> bool:
    """A stable split: the same rule is held out in every build, on every machine."""
    return int(hashlib.sha256(rule_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < HELD_OUT_SHARE


def occurrences(text: str, value: str) -> list[dict]:
    """Every occurrence of a secret value. No word boundary: a secret often touches a letter."""
    spans, start = [], text.find(value)
    while start != -1:
        spans.append({"start": start, "end": start + len(value), "label": "secret"})
        start = text.find(value, start + len(value))
    return spans


def row_label_hit(expected: list[str], spans: list[dict]) -> bool:
    wanted = {normalized(label) for label in expected}
    return any(normalized(span["label"]) in wanted for span in spans)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def provider_samples(held: bool) -> list[dict]:
    rows = []
    for name in SAMPLE_FILES:
        for sample in read_jsonl(SEATBELT / name):
            if held_out(sample["rule"]) != held:
                continue
            entities = occurrences(sample["text"], sample["value"])
            if entities:
                rows.append(
                    {
                        "text": sample["text"],
                        "entities": entities,
                        "dont_care": [],
                        "kind": "typed",
                        "rule": sample["rule"],
                    }
                )
    return rows


def false_positive_suite(case_class: str) -> list[dict]:
    """Clean rows score as typed rows with no gold; positive rows score per row label."""
    rows = [case for case in read_jsonl(SEATBELT / "false-positive-cases.jsonl") if case["class"] == case_class]
    kind = "typed" if case_class == "clean" else "row_label"
    return [
        {"text": case["text"], "entities": [], "dont_care": [], "kind": kind, "expected": case["expected"]}
        for case in rows
    ]


def generated_held_out() -> list[dict]:
    """Rows scripts/rule_secrets.py built from the held-out providers only, in its code contexts."""
    return [
        {
            "text": row["text"],
            "entities": [{"start": s["start"], "end": s["end"], "label": s["label"]} for s in row["spans"]],
            "dont_care": [],
            "kind": "typed",
        }
        for row in read_jsonl(GENERATED_HELD_OUT)
    ]


def rules_sets() -> dict[str, list[dict]]:
    return {
        "seatbelt_secrets_trained": provider_samples(held=False),
        "seatbelt_secrets_held_out": provider_samples(held=True),
        "seatbelt_clean": false_positive_suite("clean"),
        "seatbelt_positive": false_positive_suite("positive"),
        "generated_held_out": generated_held_out(),
    }
