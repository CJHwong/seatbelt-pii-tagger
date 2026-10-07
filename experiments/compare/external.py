"""External test sets that no training source shares, in the row format of sets.py.

Each row carries a `kind`, which decides how score.py reads it:
  typed:  gold marks every PII span the set knows; scored like our benchmark.
  recall: gold marks persons only, and includes public figures, so only person recall counts.
  clean:  no gold; every flag is a candidate false alarm, and a sample is reviewed by hand.
"""

import ast
import csv
import json
import random
import re
import zipfile
from pathlib import Path

from pyarrow import parquet

DATA = Path(__file__).resolve().parents[2] / "data"
SEED = 13
GRETEL_ROWS = 1_500
TAB_DOCS = 60
WIKINEWS_ROWS = 500
WIKINEWS_CHARS = 1_500
CREDENTIAL_ROWS = 1_000  # per class: true lines, then false lines

GRETEL_TYPES = {
    "first_name": "person", "last_name": "person", "name": "person",
    "email": "email", "phone_number": "phone",
    "street_address": "address", "address": "address",
    "date_of_birth": "date", "ssn": "national_id", "national_id": "national_id",
    "credit_card_number": "account_number", "account_number": "account_number",
    "license_plate": "license_plate", "user_name": "username",
    "api_key": "secret", "password": "secret", "pin": "secret", "cvv": "secret",
    "medical_record_number": "ignore", "customer_id": "ignore", "employee_id": "ignore",
    "certificate_license_number": "ignore", "health_plan_beneficiary_number": "ignore",
    "tax_id": "ignore", "url": "ignore", "bank_routing_number": "ignore",
}  # fmt: skip
# Gretel marks these too, but the guideline calls them not PII, so a flag on one is a
# false alarm and they stay unmarked: dates, times, cities, countries, companies, IPs, devices.

# CredData's true class also covers UUIDs, nonces and client IDs, which the guideline does
# not call secrets. Those lines, and CredData's ambiguous X class, are set aside.
CREDENTIAL_EXCLUDED = ("UUID", "Nonce", "Client ID")


def spans_of_values(text: str, values: list[tuple[str, str]]) -> list[dict]:
    """Every whole-word occurrence of each value, longest value first, no overlaps."""
    taken: list[dict] = []
    for value, label in sorted(values, key=lambda pair: -len(pair[0])):
        pattern = r"(?<!\w)" + re.escape(value) + r"(?!\w)"
        for match in re.finditer(pattern, text):
            span = {"start": match.start(), "end": match.end(), "label": label}
            if not any(span["start"] < t["end"] and t["start"] < span["end"] for t in taken):
                taken.append(span)
    return sorted(taken, key=lambda span: span["start"])


def bio_spans(tokens: list[str], tags: list[str], joiner: str, types: dict[str, str]) -> tuple[str, list[dict]]:
    """Join tokens into text and turn BIO tags of the mapped types into character spans."""
    text, spans, open_span = "", [], None
    for index, (token, tag) in enumerate(zip(tokens, tags, strict=True)):
        start = len(text) + (len(joiner) if index else 0)
        text += (joiner if index else "") + token
        prefix, _, kind = tag.partition("-")
        if open_span and (prefix != "I" or types.get(kind) != open_span["label"]):
            spans.append(open_span)
            open_span = None
        if prefix == "I" and open_span:
            open_span["end"] = len(text)
        elif prefix in {"B", "I"} and kind in types:
            open_span = {"start": start, "end": len(text), "label": types[kind]}
    if open_span:
        spans.append(open_span)
    return text, spans


def credential_row(
    lines: list[str], line_start: int, line_end: int, value_start: int | None, value_end: int | None
) -> dict:
    """One CredData finding as a row. Line numbers count from 1; value offsets count
    from the start of the first line and the start of the last line."""
    picked = lines[line_start - 1 : line_end]
    text = "\n".join(picked)
    entities = []
    if value_start is not None and value_end is not None:
        last_line_offset = len(text) - len(picked[-1])
        entities.append({"start": value_start, "end": last_line_offset + value_end, "label": "secret"})
    return {"text": text, "entities": entities, "dont_care": [], "kind": "typed"}


def gretel_en() -> list[dict]:
    table = parquet.read_table(DATA / "gretelai/gretel-pii-masking-en-v1/data/test-00000-of-00001.parquet").to_pylist()
    rows = []
    for record in random.Random(SEED).sample(table, GRETEL_ROWS):
        found = ast.literal_eval(record["entities"])
        values = [(e["entity"], GRETEL_TYPES[e["types"][0]]) for e in found if e["types"][0] in GRETEL_TYPES]
        spans = spans_of_values(record["text"], values)
        rows.append({
            "text": record["text"],
            "entities": [s for s in spans if s["label"] != "ignore"],
            "dont_care": [s for s in spans if s["label"] == "ignore"],
            "kind": "typed",
        })  # fmt: skip
    return rows


TITLE = re.compile(r"^(?:Mr|Mrs|Ms|Miss|Dr|Judge|Sir|Lady|Lord)\.?\s+")


def without_title(mention: dict) -> dict | None:
    """The guideline marks a name, not surname plus title. None when only a surname is left."""
    text = mention["span_text"]
    match = TITLE.match(text)
    if not match:
        return {"start": mention["start_offset"], "end": mention["end_offset"], "label": "person"}
    if len(text[match.end() :].split()) < 2:
        return None
    return {"start": mention["start_offset"] + match.end(), "end": mention["end_offset"], "label": "person"}


def tab_echr() -> list[dict]:
    """Real court judgements. One annotator per document; direct person identifiers only."""
    with zipfile.ZipFile(DATA / "ildpil/text-anonymization-benchmark/echr_test.zip") as archive:
        documents = json.loads(archive.read("echr_test.json"))
    first_annotation = {}
    for document in documents:
        first_annotation.setdefault(document["doc_id"], document)
    picked = random.Random(SEED).sample(sorted(first_annotation.values(), key=lambda d: d["doc_id"]), TAB_DOCS)
    rows = []
    for document in picked:
        direct = [
            m for m in document["entity_mentions"] if m["entity_type"] == "PERSON" and m["identifier_type"] == "DIRECT"
        ]
        persons = [span for span in map(without_title, direct) if span]
        rows.append({"text": document["text"], "entities": persons, "dont_care": [], "kind": "recall"})
    return rows


def wnut_test() -> list[dict]:
    """Real English social media. Tokens are joined with spaces, as the source keeps no text."""
    root = DATA / "tner/wnut2017/dataset"
    names = {index: name for name, index in json.loads((root / "label.json").read_text()).items()}
    rows = []
    for line in (root / "test.json").read_text().splitlines():
        record = json.loads(line)
        text, spans = bio_spans(record["tokens"], [names[t] for t in record["tags"]], " ", {"person": "person"})
        rows.append({"text": text, "entities": spans, "dont_care": [], "kind": "recall"})
    return rows


def weibo_test() -> list[dict]:
    """Real Simplified Chinese social media. Named persons only; nominal mentions are not names."""
    rows, tokens, tags = [], [], []
    for line in (DATA / "external/weibo_ner/test.conll").read_text(encoding="utf-8").splitlines() + [""]:
        if line.strip():
            character, tag = line.split("\t")
            tokens.append(character[0])
            tags.append(tag)
            continue
        if tokens:
            text, spans = bio_spans(tokens, tags, "", {"PER.NAM": "person"})
            rows.append({"text": text, "entities": spans, "dont_care": [], "kind": "recall"})
        tokens, tags = [], []
    return rows


def wikinews_zhtw() -> list[dict]:
    """Traditional Chinese news with no gold: public people, organisations and hotlines only."""
    files = sorted((DATA / "erhwenkuo/wikinews-zhtw").rglob("*.parquet"))
    articles = [a for f in files for a in parquet.read_table(f, columns=["text"]).column("text").to_pylist() if a]
    picked = random.Random(SEED).sample(articles, WIKINEWS_ROWS)
    return [{"text": article[:WIKINEWS_CHARS], "entities": [], "dont_care": [], "kind": "clean"} for article in picked]


def creddata() -> list[dict]:
    """Real source-code lines from CredData: true credentials, then false lookalikes."""
    root = DATA / "external/CredData"
    true_rows, false_rows = [], []
    for meta in sorted((root / "meta").glob("*.csv")):
        with open(meta, newline="", encoding="utf-8") as handle:
            for finding in csv.DictReader(handle):
                row = finding_row(root, finding)
                if row is not None:
                    (true_rows if row["entities"] else false_rows).append(row)
    generator = random.Random(SEED)
    return generator.sample(true_rows, CREDENTIAL_ROWS) + generator.sample(false_rows, CREDENTIAL_ROWS)


def finding_row(root: Path, finding: dict) -> dict | None:
    truth, category = finding["GroundTruth"], finding["Category"]
    if truth == "X" or any(word in category for word in CREDENTIAL_EXCLUDED):
        return None
    path = root / finding["FilePath"]
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start, end = int(finding["LineStart"]), int(finding["LineEnd"])
    if end > len(lines):
        return None
    if truth == "F":
        return {**credential_row(lines, start, end, None, None), "category": category}
    if not finding["ValueStart"] or not finding["ValueEnd"]:
        return None
    value_start, value_end = int(float(finding["ValueStart"])), int(float(finding["ValueEnd"]))
    return {**credential_row(lines, start, end, value_start, value_end), "category": category}


def external_sets() -> dict[str, list[dict]]:
    sets = {
        "gretel_en": gretel_en(),
        "tab_echr": tab_echr(),
        "wnut17": wnut_test(),
        "weibo": weibo_test(),
        "wikinews_zhtw": wikinews_zhtw(),
    }
    if (DATA / "external/CredData/data").exists():  # its own downloader fills data/, on Linux only
        sets["creddata"] = creddata()
    return sets
