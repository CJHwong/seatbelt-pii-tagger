"""Make every national ID in raw/*.reviewed.jsonl pass its check digit. Rewrites the files.

usage: python check_digits.py

Real IDs always pass their check digit, and a detector that validates it would otherwise
miss fictional values that real text never holds. Only the last character changes, and
the spacing inside the value stays as written.
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).parent
TAIWAN_LETTERS = "ABCDEFGHJKLMNPQRSTUVXYWZIO"  # the letter order sets the codes 10 to 35
TAIWAN_WEIGHTS = [1, 9, 8, 7, 6, 5, 4, 3, 2, 1]
MAINLAND_WEIGHTS = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
MAINLAND_CHECK = "10X98765432"
NATIONAL_ID_MARK = re.compile(r"\[\[national_id:(.+?)\]\]", re.DOTALL)


def taiwan_check(first_nine: str) -> str:
    """The last digit for a letter plus 8 digits (the letter, then the sex digit and 7 more)."""
    code = TAIWAN_LETTERS.index(first_nine[0]) + 10
    digits = [code // 10, code % 10] + [int(c) for c in first_nine[1:]]
    total = sum(digit * weight for digit, weight in zip(digits, TAIWAN_WEIGHTS, strict=True))
    return str((10 - total % 10) % 10)


def mainland_check(first_seventeen: str) -> str:
    total = sum(int(c) * w for c, w in zip(first_seventeen, MAINLAND_WEIGHTS, strict=True))
    return MAINLAND_CHECK[total % 11]


def fixed_id(value: str) -> str:
    """The value with a valid last character, or unchanged when its shape is neither ID."""
    compact = re.sub(r"[\s-]", "", value).upper()
    if re.fullmatch(r"[A-Z][12]\d{8}", compact):
        check = taiwan_check(compact[:9])
    elif re.fullmatch(r"\d{17}[\dX]", compact):
        check = mainland_check(compact[:17])
    else:
        return value
    last = max(i for i, c in enumerate(value) if c.isalnum())
    return value[:last] + check + value[last + 1 :]


def fixed_text(marked: str) -> str:
    return NATIONAL_ID_MARK.sub(lambda m: f"[[national_id:{fixed_id(m.group(1))}]]", marked)


def main() -> None:
    for path in sorted(HERE.glob("raw/*.reviewed.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        changed = 0
        for row in rows:
            new = fixed_text(row["text"])
            changed += new != row["text"]
            row["text"] = new
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"{path.name}: {changed} rows changed")


if __name__ == "__main__":
    main()
