"""The benchmarks the external systems publish that no other group holds, in the row format
of sets.py. Every set is report-only: it never chooses a run, a rule or a threshold.

Sets (licences in datasets.yaml; the restricted ones are for local evaluation only):
  nemotron_test:    nvidia/Nemotron-PII test split (CC-BY-4.0), English, typed, read with the
                    training label map. GLiNER-PII reports on it. Training reads only the
                    train split, but both splits come from one generator.
  pii_masking_300k: ai4privacy/pii-masking-300k English validation file (academic and
                    non-commercial only), typed. OpenAI PF and GLiNER-PII report on it. It
                    shares a generator with pii-masking-openpii-1.5m, which trains.
                    Both typed sets read as training reads them: a city, state or postcode
                    without a street stays unmarked, where argilla scores it don't-care.
  wikiann_en, wikiann_zh:       WikiANN test (licence unknown), person recall.
  multinerd_en, multinerd_zh:   MultiNERD test (CC-BY-NC-SA-4.0), person recall.
                    Redact reports name recall on both. Their gold marks public figures,
                    which the guideline does not, so only recall is scored, as with WNUT.
  kiji_en:          Kiji PII test split, English rows (Apache-2.0), typed. The OPF paper
                    (arXiv 2608.02616) reports on it. No Kiji data trains.
  gretel_finance_en: gretelai/synthetic_pii_finance_multilingual English_test (Apache-2.0),
                    typed, read with the training label map. The paper reports on it.
                    English_train trains, so this set is in-distribution.
  conll2003_en:     CoNLL-2003 test (research only), person recall. The paper reports on it.
  multiconer_en, multiconer_zh: MultiCoNER v2 test (CC-BY-4.0), person recall over its seven
                    person classes. The paper reports on it. zh is Simplified, per character.
"""

import ast
import json
import random
import sys
from functools import cache
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "scripts")]

DATA = ROOT / "data"
EXTERNAL = DATA / "external"
SEED = 13
SAMPLE_ROWS = 1_500
PERSON_TAGS = {1: "B-PER", 2: "I-PER"}  # WikiANN, MultiNERD and CoNLL-2003 share these ids
MULTICONER_PERSONS = {"Artist", "Athlete", "Cleric", "OtherPER", "Politician", "Scientist", "SportsManager"}

# The 300k labels onto the categories the training map gives pii-masking-openpii-1.5m.
PII_300K_LABELS = {
    "GIVENNAME1": "person_part", "GIVENNAME2": "person_part",
    "LASTNAME1": "person_part", "LASTNAME2": "person_part", "LASTNAME3": "person_part",
    "EMAIL": "email", "TEL": "phone", "USERNAME": "username", "PASS": "secret",
    "IDCARD": "national_id", "SOCIALNUMBER": "national_id", "PASSPORT": "passport",
    "DRIVERLICENSE": "driver_license", "BOD": "date",
    "BUILDING": "address_strong", "STREET": "address_strong", "SECADDRESS": "address_strong",
    "CITY": "address_weak", "STATE": "address_weak", "POSTCODE": "address_weak", "COUNTRY": "address_weak",
    "IP": "ignore", "GEOCOORD": "ignore", "DATE": "ignore",
    "TIME": "outside", "SEX": "outside", "TITLE": "outside", "CARDISSUER": "outside",
}  # fmt: skip

# The Kiji labels onto the same categories, following the openpii entries in type_map.yaml.
KIJI_LABELS = {
    "FIRSTNAME": "person_part", "SURNAME": "person_part",
    "EMAIL": "email", "PHONENUMBER": "phone", "URL": "url", "USERNAME": "username",
    "PASSWORD": "secret", "SECURITYTOKEN": "secret",
    "SSN": "national_id", "NATIONALID": "national_id", "IDCARDNUM": "national_id",
    "PASSPORTID": "passport", "DRIVERLICENSENUM": "driver_license", "LICENSEPLATENUM": "license_plate",
    "IBAN": "account_number", "CREDITCARDNUMBER": "account_number", "DATEOFBIRTH": "date",
    "BUILDINGNUM": "address_strong", "STREET": "address_strong",
    "CITY": "address_weak", "STATE": "address_weak", "REGION": "address_weak", "ZIP": "address_weak",
    "COUNTRY": "address_weak",
    "TAXNUM": "ignore",
    "AGE": "outside", "TITLE": "outside", "COMPANYNAME": "outside", "ORGANIZATION": "outside",
}  # fmt: skip


def mapped_row(text: str, spans: list[dict], labels: dict[str, str], language: str | None) -> dict | None:
    from build_training_set import process

    config = {"labels": labels, "languages": None, "keep_chinese": True}
    record, _, _ = process({"text": text, "spans": spans, "language": language}, config)
    if record is None:
        return None
    return {"text": record["text"], "entities": record["entities"], "dont_care": record["ignore"], "kind": "typed"}


def pii_300k_row(text: str, spans: list[dict]) -> dict | None:
    """None for a row the training checks drop: 2 of the 7,946, a bad offset and a script outside ours."""
    return mapped_row(text, [{k: s[k] for k in ("start", "end", "label")} for s in spans], PII_300K_LABELS, "en")


def kiji_row(text: str, spans: list[dict]) -> dict | None:
    return mapped_row(text, [{k: s[k] for k in ("start", "end", "label")} for s in spans], KIJI_LABELS, "en")


def person_row(tokens: list[str], tags: list[str], joiner: str) -> dict:
    from external import bio_spans

    text, spans = bio_spans(tokens, tags, joiner, {"PER": "person"})
    return {"text": text, "entities": spans, "dont_care": [], "kind": "recall"}


def names_row(tokens: list[str], tags: list[int], joiner: str) -> dict:
    return person_row(tokens, [PERSON_TAGS.get(t, "O") for t in tags], joiner)


def multiconer_tag(tag: str) -> str:
    """B-Politician becomes B-PER; every class outside the person group becomes O."""
    prefix, _, kind = tag.partition("-")
    return f"{prefix}-PER" if kind in MULTICONER_PERSONS else "O"


def multiconer_rows(lines: list[str], joiner: str) -> list[dict]:
    """The CoNLL file: a "# id" line, then one "token _ _ TAG" line per token, then a blank line."""
    rows, tokens, tags = [], [], []
    for line in [*lines, ""]:
        fields = line.split()
        if fields and not line.startswith("# id"):
            tokens.append(fields[0])
            tags.append(multiconer_tag(fields[-1]))
        elif not fields and tokens:
            rows.append(person_row(tokens, tags, joiner))
            tokens, tags = [], []
    return rows


def sample(rows: list[dict]) -> list[dict]:
    return random.Random(SEED).sample(rows, min(SAMPLE_ROWS, len(rows)))


PII_300K = EXTERNAL / "pii-masking-300k/data/validation/1english_openpii_8k.jsonl"


def nemotron_test() -> list[dict]:
    import yaml
    from pyarrow import parquet

    labels = yaml.safe_load((ROOT / "type_map.yaml").read_text())["datasets"]["nvidia/Nemotron-PII"]["labels"]
    table = parquet.read_table(DATA / "nvidia/Nemotron-PII/data/test-00000-of-00001.parquet", columns=["text", "spans"])
    records = list(zip(table.column("text").to_pylist(), table.column("spans").to_pylist(), strict=True))
    rows = [mapped_row(text, ast.literal_eval(spans) if spans else [], labels, None) for text, spans in sample(records)]
    return [row for row in rows if row is not None]


def pii_masking_300k() -> list[dict]:
    records = [json.loads(line) for line in PII_300K.read_text(encoding="utf-8").splitlines()]
    rows = [pii_300k_row(r["source_text"], r["privacy_mask"]) for r in records]
    return sample([row for row in rows if row is not None])


def wikiann(language: str) -> list[dict]:
    from pyarrow import parquet

    table = parquet.read_table(
        EXTERNAL / f"wikiann/{language}/test-00000-of-00001.parquet", columns=["tokens", "ner_tags"]
    )
    joiner = "" if language == "zh" else " "
    return [names_row(r["tokens"], r["ner_tags"], joiner) for r in sample(table.to_pylist())]


def multinerd(language: str) -> list[dict]:
    path = EXTERNAL / f"multinerd/test/test_{language}.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    joiner = "" if language == "zh" else " "
    return [names_row(r["tokens"], r["ner_tags"], joiner) for r in sample(records)]


def kiji_en() -> list[dict]:
    from pyarrow import parquet

    table = parquet.read_table(EXTERNAL / "kiji-pii-training-data/data/test-00000-of-00001.parquet")
    records = [r for r in table.to_pylist() if r["language"] == "English"]
    rows = [kiji_row(r["text"], r["privacy_mask"]) for r in records]
    return sample([row for row in rows if row is not None])


def gretel_finance_en() -> list[dict]:
    import yaml
    from pyarrow import parquet

    labels = yaml.safe_load((ROOT / "type_map.yaml").read_text())["datasets"][
        "gretelai/synthetic_pii_finance_multilingual"
    ]["labels"]
    path = DATA / "gretelai/synthetic_pii_finance_multilingual/data/English_test-00000-of-00001.parquet"
    table = parquet.read_table(path, columns=["generated_text", "pii_spans"])
    records = zip(table.column("generated_text").to_pylist(), table.column("pii_spans").to_pylist(), strict=True)
    rows = [mapped_row(text, json.loads(spans) if spans else [], labels, "en") for text, spans in records]
    return sample([row for row in rows if row is not None])


def conll2003_en() -> list[dict]:
    from pyarrow import parquet

    table = parquet.read_table(EXTERNAL / "conll2003/conll2003/test/0000.parquet", columns=["tokens", "ner_tags"])
    return [names_row(r["tokens"], r["ner_tags"], " ") for r in sample(table.to_pylist())]


def multiconer(language: str) -> list[dict]:
    folder = {"en": "EN-English", "zh": "ZH-Chinese"}[language]
    lines = (EXTERNAL / f"multiconer_v2/{folder}/{language}_test.conll").read_text(encoding="utf-8").splitlines()
    return sample(multiconer_rows(lines, "" if language == "zh" else " "))


@cache
def published_sets() -> dict[str, list[dict]]:
    sets = {"nemotron_test": nemotron_test()}
    if PII_300K.exists():  # sharing its scores needs written permission, so most clones lack it
        sets["pii_masking_300k"] = pii_masking_300k()
    return sets | {
        "wikiann_en": wikiann("en"),
        "wikiann_zh": wikiann("zh"),
        "multinerd_en": multinerd("en"),
        "multinerd_zh": multinerd("zh"),
        "kiji_en": kiji_en(),
        "gretel_finance_en": gretel_finance_en(),
        "conll2003_en": conll2003_en(),
        "multiconer_en": multiconer("en"),
        "multiconer_zh": multiconer("zh"),
    }
