"""Turn the reviewed rows in raw/*.reviewed.jsonl into benchmark.jsonl with character offsets.

Writers mark spans inline as [[type:value]], so no offset is ever counted by hand.
Any malformed mark stops the build. The output row format matches the prepared
training rows: {"id", "register", "text", "entities", "ignore"}.
"""

import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
TYPES = {
    "person", "email", "phone", "address", "date", "url", "account_number", "secret",
    "national_id", "passport", "license_plate", "username", "nhi_card", "company_id",
    "driver_license", "medical_license", "military_id", "household_no",
}  # fmt: skip
MARK = re.compile(r"\[\[(\w+):(.+?)\]\]", re.DOTALL)


def parse_marked(marked: str) -> tuple[str, list[dict]]:
    """Plain text plus spans. Raises ValueError on an unknown type or a stray bracket."""
    pieces, spans, cursor, length = [], [], 0, 0
    for match in MARK.finditer(marked):
        label, value = match.group(1), match.group(2)
        if label not in TYPES and label != "ignore":
            raise ValueError(f"unknown type {label!r} in: {marked[:60]}")
        if "[[" in value:
            raise ValueError(f"nested mark in: {marked[:60]}")
        literal = marked[cursor : match.start()]
        pieces.append(literal)
        length += len(literal)
        spans.append({"start": length, "end": length + len(value), "label": label})
        pieces.append(value)
        length += len(value)
        cursor = match.end()
    pieces.append(marked[cursor:])
    text = "".join(pieces)
    if "[[" in text or "]]" in text:
        raise ValueError(f"unmatched bracket in: {marked[:60]}")
    return text, spans


def build_row(raw: dict) -> dict:
    text, spans = parse_marked(raw["text"])
    return {
        "id": raw["id"],
        "register": raw["register"],
        "text": text,
        "entities": [s for s in spans if s["label"] != "ignore"],
        "ignore": [{"start": s["start"], "end": s["end"]} for s in spans if s["label"] == "ignore"],
    }


def main() -> None:
    rows = []
    for path in sorted((HERE / "raw").glob("*.reviewed.jsonl")):
        with open(path, encoding="utf-8") as handle:
            rows += [build_row(json.loads(line)) for line in handle if line.strip()]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate ids")
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    (HERE / "benchmark.jsonl").write_text(body, encoding="utf-8")
    print(f"{len(rows)} rows, sha256 {hashlib.sha256(body.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
