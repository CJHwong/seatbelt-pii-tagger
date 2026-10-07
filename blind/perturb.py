"""Label-preserving copies of our benchmark that test robustness, not new content.

usage: python perturb.py   -> perturbed.jsonl

Each variant keeps only the rows it changed, and writes the same rows unchanged as
VARIANT_original, so a score delta comes from the perturbation alone:
  fullwidth: every ASCII digit becomes its full-width form, as IME input produces.
  spaced:    digit runs of 6 or more inside identifier spans get a space every 4 digits.
  ocr:       lookalike swaps inside spans (0 to O, 1 to l, 5 to S, 8 to B), as a bad scan gives.
  long:      30 rows joined into one document, so spans sit far past the first chunk.
"""

import hashlib
import json
import random
import re
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).parent
BENCHMARK = HERE.parent / "benchmark" / "benchmark.jsonl"
NUMERIC = {
    "phone", "account_number", "national_id", "passport", "nhi_card", "company_id",
    "driver_license", "medical_license", "military_id", "household_no", "ignore",
}  # fmt: skip
OCR_SWAPS = {"0": "O", "1": "l", "5": "S", "8": "B"}
OCR_RATE = 0.3
LONG_ROWS = 30
SEED = 13

Transform = Callable[[str, str | None], str]


def fullwidth(text: str, _label: str | None) -> str:
    return "".join(chr(ord(c) + 0xFEE0) if "0" <= c <= "9" else c for c in text)


def spaced(text: str, label: str | None) -> str:
    if label not in NUMERIC:
        return text
    return re.sub(r"\d{6,}", lambda m: " ".join(re.findall(r"\d{1,4}", m.group())), text)


def ocr(text: str, label: str | None) -> str:
    if label is None:
        return text
    generator = random.Random(hashlib.sha256(text.encode()).digest())
    return "".join(OCR_SWAPS[c] if c in OCR_SWAPS and generator.random() < OCR_RATE else c for c in text)


def rewrite(row: dict, inside: Transform, outside: Transform) -> dict:
    """Transform each segment of the text; spans follow their segments to the new offsets."""
    marks = [(e["start"], e["end"], e["label"]) for e in row["entities"]]
    marks += [(i["start"], i["end"], "ignore") for i in row["ignore"]]
    pieces, entities, ignore, cursor = [], [], [], 0
    for start, end, label in sorted(marks):
        pieces.append(outside(row["text"][cursor:start], None))
        new_start = sum(map(len, pieces))
        pieces.append(inside(row["text"][start:end], label))
        span = {"start": new_start, "end": new_start + len(pieces[-1])}
        (ignore.append(span) if label == "ignore" else entities.append({**span, "label": label}))
        cursor = end
    pieces.append(outside(row["text"][cursor:], None))
    return {**row, "text": "".join(pieces), "entities": entities, "ignore": ignore}


def joined(rows: list[dict], separator: str) -> dict:
    text, entities, ignore = "", [], []
    for index, row in enumerate(rows):
        offset = len(text) + (len(separator) if index else 0)
        text += (separator if index else "") + row["text"]
        entities += [{**e, "start": e["start"] + offset, "end": e["end"] + offset} for e in row["entities"]]
        ignore += [{"start": i["start"] + offset, "end": i["end"] + offset} for i in row["ignore"]]
    return {"id": "+".join(r["id"] for r in rows), "text": text, "entities": entities, "ignore": ignore}


def variants(rows: list[dict]) -> list[dict]:
    keep = lambda text, _label: text
    edits = {"fullwidth": (fullwidth, fullwidth), "spaced": (spaced, keep), "ocr": (ocr, keep)}
    out = []
    for name, (inside, outside) in edits.items():
        for row in rows:
            changed = rewrite(row, inside, outside)
            if changed["text"] != row["text"]:
                out += [{**changed, "variant": name}, {**row, "variant": f"{name}_original"}]
    shuffled = rows[:]
    random.Random(SEED).shuffle(shuffled)
    for start in range(0, len(shuffled) - LONG_ROWS + 1, LONG_ROWS):
        group = shuffled[start : start + LONG_ROWS]
        out.append({**joined(group, "\n\n"), "variant": "long"})
        out += [{**row, "variant": "long_original"} for row in group]
    return out


def main() -> None:
    rows = [json.loads(line) for line in BENCHMARK.read_text(encoding="utf-8").splitlines()]
    out = variants(rows)
    with open(HERE / "perturbed.jsonl", "w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in out)
    counts: dict[str, int] = {}
    for row in out:
        counts[row["variant"]] = counts.get(row["variant"], 0) + 1
    print(counts)


if __name__ == "__main__":
    main()
