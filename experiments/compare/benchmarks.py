"""The benchmarks other systems report, in the row format of sets.py, plus their scoring.

usage:
  uv run --with faker --with pyarrow python benchmarks.py build-spy    freeze the SPY sample
  uv run --with pyarrow python benchmarks.py score                     score the case suites

Sets:
  spy:            SPY (mks-logic/SPY, CC-BY-4.0), legal and medical English, typed.
  argilla:        argilla/textcat-tokencat-pii-per-domain, 1% of ai4privacy/pii-masking-200k,
                  English, typed. Local evaluation only; see datasets.yaml for the licence.
  seatbelt_*:     the agent-seatbelt hook suites. Each row is a case scored like the suite's
                  own harness: a positive case passes when every expected label is predicted,
                  a clean case fails on any label, an ambiguous case is only counted.
"""

import importlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data" / "external"
SPY = DATA / "SPY"
SPY_FILLED = SPY / "spy_filled_seed13.jsonl"
ARGILLA = DATA / "argilla-pii" / "train.parquet"
SUITE = DATA / "agent-seatbelt" / "hooks-pii" / "tests"
SEED = 13
SAMPLE_ROWS = 1_500
FOLDED_INTO_ACCOUNT = {
    "national_id", "passport", "nhi_card", "company_id", "driver_license",
    "medical_license", "military_id", "household_no",
}  # fmt: skip
HOOK_LABELS = {
    "private_person": "person", "private_email": "email", "private_phone": "phone",
    "private_address": "address", "private_url": "url", "private_date": "date",
}  # fmt: skip

# SPY marks every URL as PII, but most of its Faker URLs are shop and company sites, which
# the guideline does not mark, so they count neither way.
SPY_TYPES = {"NAME": "person", "EMAIL": "email", "PHONE_NUM": "phone", "ADDRESS": "address", "USERNAME": "username"}
ARGILLA_TYPES = {
    "EMAIL": "email", "PHONENUMBER": "phone", "USERNAME": "username", "DOB": "date",
    "PASSWORD": "secret", "PIN": "secret", "CREDITCARDCVV": "secret",
    "ACCOUNTNUMBER": "account_number", "CREDITCARDNUMBER": "account_number", "IBAN": "account_number",
    "SSN": "national_id", "VEHICLEVRM": "license_plate",
}  # fmt: skip
ARGILLA_NAME_PARTS = {"FIRSTNAME", "MIDDLENAME", "LASTNAME"}
ARGILLA_ADDRESS_PARTS = {"STREET", "BUILDINGNUMBER", "SECONDARYADDRESS", "CITY", "STATE", "ZIPCODE", "COUNTY"}
ARGILLA_STREET_LEVEL = {"STREET", "BUILDINGNUMBER"}
ARGILLA_IGNORED = {
    "URL", "MASKEDNUMBER", "ACCOUNTNAME", "IPV4", "IPV6", "IP", "MAC", "PHONEIMEI", "BITCOINADDRESS",
    "ETHEREUMADDRESS", "LITECOINADDRESS", "VEHICLEVIN", "BIC", "USERAGENT", "NEARBYGPSCOORDINATE",
}  # fmt: skip
# Every other Argilla label (prefix, job, company, amount, gender, age, date, time) is not PII
# under the guideline, so it stays unmarked and a flag on it is a false alarm.


def filled_spy_row(tokens: list[str], tags: list[str], profile: dict) -> dict:
    """SPY tokens carry their own trailing space; a placeholder token is replaced by its value."""
    text, entities, dont_care = "", [], []
    for token, tag in zip(tokens, tags, strict=True):
        if not tag.startswith("B-"):
            text += token
            continue
        kind = tag[2:]
        value, label = profile[kind] if isinstance(profile[kind], tuple) else (profile[kind], SPY_TYPES.get(kind))
        start = len(text) + token.index("<")
        text += token.replace(f"<{kind}>", value)
        span = {"start": start, "end": start + len(value)}
        (entities.append({**span, "label": label}) if label else dont_care.append(span))
    return {"text": text, "entities": entities, "dont_care": dont_care, "kind": "typed"}


def spy_profile(faker) -> dict:
    """One value per entity type for a row, as SPY's loader makes one profile per row.
    ID_NUM keeps only SPY's generators that map to one of our types."""
    id_generators = [
        (faker.ssn, "national_id"),
        (faker.credit_card_number, "account_number"),
        (faker.iban, "account_number"),
        (faker.bban, "account_number"),
    ]
    generate, label = faker.random_element(id_generators)
    return {
        "NAME": faker.name(),
        "EMAIL": faker.random_element([faker.ascii_email, faker.ascii_free_email])(),
        "URL": faker.uri(deep=faker.random_element([1, 2])),
        "PHONE_NUM": faker.phone_number(),
        "ID_NUM": (generate(), label),
        "ADDRESS": faker.street_address(),
        "USERNAME": faker.user_name(),
    }


def build_spy() -> None:
    from importlib.metadata import version

    from faker import Faker

    Faker.seed(SEED)
    faker = Faker()
    records = [
        json.loads(line)
        for name in ("legal_questions", "medical_consultations")
        for line in (SPY / f"{name}_placeholders.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    picked = random.Random(SEED).sample(records, SAMPLE_ROWS)
    rows = [filled_spy_row(r["tokens"], r["ent_tags"], spy_profile(faker)) for r in picked]
    SPY_FILLED.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(f"{len(rows)} SPY rows written with Faker {version('faker')}")


def spy() -> list[dict]:
    return [json.loads(line) for line in SPY_FILLED.read_text(encoding="utf-8").splitlines()]


def grouped(spans: list[dict], text: str, members: set[str], separators: str) -> list[list[dict]]:
    """Runs of member spans that only the separator characters divide."""
    runs: list[list[dict]] = []
    for span in sorted(spans, key=lambda s: s["start"]):
        if span["label"] not in members:
            continue
        gap = text[runs[-1][-1]["end"] : span["start"]] if runs else None
        if gap is not None and all(c in separators for c in gap):
            runs[-1].append(span)
        else:
            runs.append([span])
    return runs


def argilla_row(text: str, spans: list[dict]) -> dict:
    entities, dont_care = [], []
    for run in grouped(spans, text, ARGILLA_NAME_PARTS, " "):
        entities.append({"start": run[0]["start"], "end": run[-1]["end"], "label": "person"})
    for run in grouped(spans, text, ARGILLA_ADDRESS_PARTS, " ,"):
        span = {"start": run[0]["start"], "end": run[-1]["end"]}
        street_level = any(s["label"] in ARGILLA_STREET_LEVEL for s in run)
        (entities.append({**span, "label": "address"}) if street_level else dont_care.append(span))
    for span in spans:
        if span["label"] in ARGILLA_TYPES:
            entities.append({"start": span["start"], "end": span["end"], "label": ARGILLA_TYPES[span["label"]]})
        elif span["label"] in ARGILLA_IGNORED:
            dont_care.append({"start": span["start"], "end": span["end"]})
    entities.sort(key=lambda s: s["start"])
    return {"text": text, "entities": entities, "dont_care": dont_care, "kind": "typed"}


def argilla() -> list[dict]:
    from pyarrow import parquet

    table = parquet.read_table(ARGILLA, columns=["source-text", "pii.suggestion"]).to_pylist()
    picked = random.Random(SEED).sample(table, min(SAMPLE_ROWS, len(table)))
    return [argilla_row(record["source-text"], record["pii.suggestion"] or []) for record in picked]


def suite_cases(module_name: str, attribute: str) -> list[dict]:
    sys.path.insert(0, str(SUITE))
    cases = getattr(importlib.import_module(module_name), attribute)
    return [{**case, "entities": [], "dont_care": [], "kind": "cases"} for case in cases]


def prompt_cases() -> list[dict]:
    rows = [json.loads(line) for line in (SUITE / "test-cases.jsonl").read_text().splitlines() if line.strip()]
    return [
        {
            "class": "positive",
            "expected": [r["category"]],
            "text": r["prompt"],
            "entities": [],
            "dont_care": [],
            "kind": "cases",
        }
        for r in rows
    ]


def benchmark_sets() -> dict[str, list[dict]]:
    return {
        "spy": spy(),
        "argilla": argilla(),
        "seatbelt_test_cases": prompt_cases(),
        "seatbelt_holdout": suite_cases("holdout_cases", "HOLDOUT_CASES"),
        "seatbelt_final_holdout": suite_cases("final_holdout_cases", "FINAL_HOLDOUT_CASES"),
        "seatbelt_challenge": suite_cases("challenge_cases", "CASES"),
    }


def case_label(label: str) -> str:
    label = HOOK_LABELS.get(label, label)
    return "account_number" if label in FOLDED_INTO_ACCOUNT else label


def score_cases(cases: list[dict], predictions: list[list[dict]]) -> dict:
    """The suite harness rules. Robustness cases are positives written to be hard, so they
    count apart, with the positive rule."""
    report = {
        "positive": Counter(),
        "robustness": Counter(),
        "clean": Counter(),
        "ambiguous": Counter(),
    }
    for case, spans in zip(cases, predictions, strict=True):
        labels = {case_label(s["label"]) for s in spans}
        kind = case["class"]
        report[kind]["total"] += 1
        if kind in {"positive", "robustness"}:
            report[kind]["passed"] += {case_label(e) for e in case["expected"]} <= labels
        else:
            report[kind]["flagged"] += bool(labels)
    return {kind: dict(counts) for kind, counts in report.items() if counts}


def case_report(systems: list[str]) -> str:
    sets = {name: rows for name, rows in benchmark_sets().items() if rows and rows[0]["kind"] == "cases"}
    lines = ["| suite | measure | " + " | ".join(systems) + " |", "|---|---|" + "---|" * len(systems)]
    reports = {}
    for system in systems:
        by_set: dict[str, dict[int, list[dict]]] = {}
        for line in (HERE / "runs" / "benchmarks" / f"{system}.jsonl").read_text().splitlines():
            record = json.loads(line)
            by_set.setdefault(record["set"], {})[record["index"]] = record["spans"]
        reports[system] = {
            name: score_cases(rows, [by_set[name].get(i, []) for i in range(len(rows))]) for name, rows in sets.items()
        }
    for name in sets:
        for kind, measure in (
            ("positive", "passed"),
            ("robustness", "passed"),
            ("clean", "flagged"),
            ("ambiguous", "flagged"),
        ):
            if kind not in reports[systems[0]][name]:
                continue
            cells = [f"{reports[s][name][kind].get(measure, 0)} of {reports[s][name][kind]['total']}" for s in systems]
            lines.append(f"| {name} | {kind} {measure} | " + " | ".join(cells) + " |")
    (HERE / "runs" / "benchmarks" / "cases.json").write_text(json.dumps(reports, indent=1))
    return "\n".join(lines) + "\n"


def main() -> None:
    if sys.argv[1] == "build-spy":
        build_spy()
        return
    systems = [s for s in sys.argv[2:]] or [
        p.stem for p in sorted((HERE / "runs" / "benchmarks").glob("*.jsonl")) if not p.stem.endswith(".flags")
    ]
    report = case_report(systems)
    (HERE / "runs" / "benchmarks" / "cases.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
