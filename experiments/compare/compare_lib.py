"""Scoring that treats every system the same, whatever its label set.

Every label maps into the Privacy Filter's 8 categories, the only set all systems share.
Our ID types fold into account_number, as the Privacy Filter and Redact fold them.
`username` and `license_plate` have no category there, so typed scores leave them out,
and `date` is left out because the systems disagree on which dates are PII.
"""

import hashlib
import re
from collections import Counter
from functools import cache
from pathlib import Path

WHITESPACE = re.compile(r"\s+")
LOCAL_SYSTEMS = Path(__file__).parent / "systems.local.yaml"


@cache
def local_systems() -> dict:
    """Extra runs, ensembles, exports and scored systems from LOCAL_SYSTEMS, or {} without it."""
    if not LOCAL_SYSTEMS.exists():
        return {}
    import yaml

    return yaml.safe_load(LOCAL_SYSTEMS.read_text(encoding="utf-8")) or {}


HOOK_TO_OURS = {
    "private_person": "person",
    "private_email": "email",
    "private_phone": "phone",
    "private_address": "address",
    "private_url": "url",
    "private_date": "date",
    "account_number": "account_number",
    "secret": "secret",
}
FOLDED_INTO_ACCOUNT = {
    "national_id", "passport", "nhi_card", "company_id", "driver_license",
    "medical_license", "military_id", "household_no",
}  # fmt: skip
NO_SHARED_CATEGORY = {"username", "license_plate"}


def normalized(label: str) -> str:
    return HOOK_TO_OURS.get(label, label)


def category(label: str) -> str | None:
    """The shared category for a label in our types, or None when it has none or is a date."""
    if label in NO_SHARED_CATEGORY or label == "date":
        return None
    return "account_number" if label in FOLDED_INTO_ACCOUNT else label


def overlaps(first: dict, second: dict) -> bool:
    return first["start"] < second["end"] and second["start"] < first["end"]


def outside(spans: list[dict], dont_care: list[dict]) -> list[dict]:
    return [s for s in spans if not any(overlaps(s, d) for d in dont_care)]


def row_counts(gold: list[dict], predicted: list[dict]) -> Counter:
    """Counts for one row: typed on the shared categories (strict and overlap), and untyped overlap.

    Dates are removed on both sides before any count, so a system that tags every date
    is neither punished nor rewarded for it.
    """
    gold = [g for g in gold if g["label"] != "date"]
    predicted = [p for p in predicted if p["label"] != "date"]
    counts = Counter()
    typed_gold = [{**g, "category": category(g["label"])} for g in gold if category(g["label"])]
    typed_predicted = [{**p, "category": category(p["label"])} for p in predicted if category(p["label"])]
    exact = {(g["start"], g["end"], g["category"]) for g in typed_gold}
    counts["strict_tp"] = sum((p["start"], p["end"], p["category"]) in exact for p in typed_predicted)
    counts["strict_gold"], counts["strict_predicted"] = len(typed_gold), len(typed_predicted)
    same = lambda a, b: a["category"] == b["category"] and overlaps(a, b)
    counts["typed_gold_hit"] = sum(any(same(g, p) for p in typed_predicted) for g in typed_gold)
    counts["typed_predicted_hit"] = sum(any(same(p, g) for g in typed_gold) for p in typed_predicted)
    counts["any_gold"], counts["any_predicted"] = len(gold), len(predicted)
    counts["any_gold_hit"] = sum(any(overlaps(g, p) for p in predicted) for g in gold)
    counts["any_predicted_hit"] = sum(any(overlaps(p, g) for g in gold) for p in predicted)
    return counts


def f1(true_for_precision: int, predicted: int, true_for_recall: int, gold: int) -> dict:
    precision = true_for_precision / predicted if predicted else 0.0
    recall = true_for_recall / gold if gold else 0.0
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": score}


def summary(counts: Counter) -> dict:
    return {
        "typed_strict": f1(counts["strict_tp"], counts["strict_predicted"], counts["strict_tp"], counts["strict_gold"]),
        "typed_overlap": f1(
            counts["typed_predicted_hit"], counts["strict_predicted"], counts["typed_gold_hit"], counts["strict_gold"]
        ),
        "any_overlap": f1(
            counts["any_predicted_hit"], counts["any_predicted"], counts["any_gold_hit"], counts["any_gold"]
        ),
    }


def text_key(text: str) -> str:
    """A text's identity for the leak check: whitespace-free, so a reflowed copy still matches."""
    return hashlib.sha256(WHITESPACE.sub("", text).encode()).hexdigest()


def mark_leaked(sets: dict[str, list[dict]], leaked: dict[str, list[str]]) -> dict[str, list[dict]]:
    """Copies of the rows, with `leaked` set on each one the leak check found in training."""
    marked = {}
    for name, rows in sets.items():
        keys = set(leaked.get(name, []))
        marked[name] = [{**row, "leaked": True} if keys and text_key(row["text"]) in keys else row for row in rows]
    return marked


def unleaked(rows: list[dict], predictions: dict[int, list]) -> tuple[list[dict], dict[int, list]]:
    """The rows the scorer may count, and their predictions under the new row numbers."""
    kept = [index for index, row in enumerate(rows) if not row.get("leaked")]
    return [rows[index] for index in kept], {
        new: predictions[old] for new, old in enumerate(kept) if old in predictions
    }


def release_run() -> str:
    """The run the release is: `release`, the name reproduce.sh trains, unless LOCAL_SYSTEMS names another."""
    return local_systems().get("release_run", "release")
