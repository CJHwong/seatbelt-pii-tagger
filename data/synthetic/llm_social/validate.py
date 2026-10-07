"""Check LLM-written rows before anyone trusts them.

usage: python3 validate.py FILE...   -> one line per bad row, then a summary; exit 1 on any bad row

Each line is {"id", "register", "script", "text"}, with spans marked inline as [[type:value]]
under benchmark/GUIDELINE.md. The parse is the benchmark's own, so the rules match.
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmark"))
from build import parse_marked

SCRIPTS = {"zh-Hant", "zh-Hans", "latin"}
FIELDS = {"id", "register", "script", "text"}


def problems(raw: dict) -> list[str]:
    found = []
    if set(raw) != FIELDS:
        found.append(f"fields {sorted(raw)} are not {sorted(FIELDS)}")
    if raw.get("script") not in SCRIPTS:
        found.append(f"script {raw.get('script')!r} is not one of {sorted(SCRIPTS)}")
    try:
        text, spans = parse_marked(raw.get("text", ""))
    except ValueError as error:
        return found + [str(error)]
    for span in spans:
        value = text[span["start"] : span["end"]]
        if value != value.strip():
            found.append(f"{span['label']} value has edge spaces: {value!r}")
    if not text.strip():
        found.append("empty text")
    return found


def main() -> None:
    bad, rows, labels, ids = 0, 0, Counter(), Counter()
    for path in sys.argv[1:]:
        for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            rows += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                bad += 1
                print(f"{path}:{number}: not JSON: {error}")
                continue
            ids[raw.get("id")] += 1
            found = problems(raw)
            if found:
                bad += 1
                print(f"{path}:{number}: {raw.get('id')}: {'; '.join(found)}")
                continue
            labels.update(span["label"] for span in parse_marked(raw["text"])[1])
    duplicates = [key for key, count in ids.items() if count > 1]
    if duplicates:
        bad += len(duplicates)
        print(f"duplicate ids: {duplicates[:20]}")
    print(f"{rows} rows, {bad} bad, spans by type: {dict(labels.most_common())}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
