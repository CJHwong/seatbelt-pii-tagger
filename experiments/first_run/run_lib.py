"""Pure helpers for the first 18-type training run: loss masking and benchmark scoring."""

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vocab_prune"))

from prune_lib import span_counts

# tw-PII-bench label to our type. The bench's own expected_model_label folds the Taiwanese
# IDs into account_number; our taxonomy keeps them apart, so it maps from the raw label.
TW_BENCH_TYPES = {
    "account_number": "account_number",
    "private_address": "address",
    "private_date": "date",
    "private_email": "email",
    "private_person": "person",
    "private_phone": "phone",
    "private_url": "url",
    "secret": "secret",
    "tw_company_id": "company_id",
    "tw_driver_license": "driver_license",
    "tw_household_no": "household_no",
    "tw_license_plate": "license_plate",
    "tw_line_id": "username",
    "tw_medical_license": "medical_license",
    "tw_military_id": "military_id",
    "tw_national_id": "national_id",
    "tw_nhi_card": "nhi_card",
    "tw_passport": "passport",
    "tw_ptt_id": "username",
}
BIRTH_CUE = re.compile(r"生日|出生|生於|壽星|[Bb]irth|DOB")
CUE_BEFORE, CUE_AFTER = 12, 6


def is_birth_date(text: str, start: int, end: int) -> bool:
    """Our date type covers birth dates only. The bench tags every private date, so a cue
    word next to the span decides which of its dates our rule covers."""
    return BIRTH_CUE.search(text[max(0, start - CUE_BEFORE) : end + CUE_AFTER]) is not None


def bench_gold(text: str, spans: list[dict]) -> tuple[list[dict], list[dict]]:
    """Gold entities in our types, plus the bench dates our rule leaves out.

    A left-out date is neither a miss nor a false positive: the scorer drops predictions
    that overlap it. Leaving it gold would punish the model for following the taxonomy.
    """
    entities, dont_care = [], []
    for span in spans:
        label = TW_BENCH_TYPES[span["label"]]
        if label == "date" and not is_birth_date(text, span["start"], span["end"]):
            dont_care.append({"start": span["start"], "end": span["end"]})
            continue
        entities.append({"start": span["start"], "end": span["end"], "label": label})
    return entities, dont_care


def drop_overlapping(predicted: list[dict], dont_care: list[dict]) -> list[dict]:
    return [p for p in predicted if not any(p["start"] < d["end"] and d["start"] < p["end"] for d in dont_care)]


def drop_name_repeats(predicted: list[dict], text: str, gold: list[dict]) -> tuple[list[dict], int]:
    """Set aside person predictions that repeat a gold name elsewhere in the row.

    The bench labels a name at some mentions only, and leaves later full or partial
    mentions (a surname, a given name) unlabelled. A redactor must mask every mention,
    so those predictions are neither hits nor false positives. Returns (kept, set aside).
    """
    names = {text[g["start"] : g["end"]] for g in gold if g["label"] == "person"}
    kept = []
    for span in predicted:
        value = text[span["start"] : span["end"]]
        repeats = span["label"] == "person" and any(value in name or name in value for name in names)
        if repeats and not any(span["start"] < g["end"] and g["start"] < span["end"] for g in gold):
            continue
        kept.append(span)
    return kept, len(predicted) - len(kept)


def mask_ignored(tags: list[str | None], token_spans: list, ignore: list[dict]) -> list[str | None]:
    """None, so the loss skips it, for every token inside an ignore span."""
    return [
        None if span and any(span[0] < i["end"] and i["start"] < span[1] for i in ignore) else tag
        for tag, span in zip(tags, token_spans, strict=True)
    ]


def per_type_counts(gold: list[dict], predicted: list[dict]) -> dict[str, Counter]:
    labels = {span["label"] for span in gold} | {span["label"] for span in predicted}
    return {
        label: span_counts(
            [g for g in gold if g["label"] == label], [p for p in predicted if p["label"] == label], typed=True
        )
        for label in labels
    }


def script_bucket(script: str) -> str:
    """Sampling bucket. 'zh' has no Traditional markers, and Simplified sources dominate it."""
    return {"latin": "latin", "zh-Hant": "zh-Hant"}.get(script, "zh-Hans")


def length_batches(lengths: list[int], batch_size: int, picker) -> list[list[int]]:
    """Cut indices into batches of similar length, in random batch order.

    A batch pads to its longest row. Random batches were 40% real tokens, so sorting
    a pool by length before cutting it made each step 2.6 times faster on Apple silicon.
    """
    order = sorted(range(len(lengths)), key=lengths.__getitem__)
    batches = [order[start : start + batch_size] for start in range(0, len(order), batch_size)]
    picker.shuffle(batches)
    return batches
