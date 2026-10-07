"""Pure helpers that turn source-dataset spans into the 18-type training spans.

Every source label maps to one of four kinds, set in `type_map.yaml`:
  a type name        one of the 18 decided types
  person_part        a name fragment, joined with adjacent fragments into `person`
  address_strong     a street-level address part, joined into `address`
  address_weak       city, state, country: `address` only when joined to a strong part
  ignore             masked out of the loss: a real identifier the list leaves out
  outside            trained as not-PII: public or non-identifying
"""

import hashlib
import re

TYPES = (
    "person",
    "email",
    "phone",
    "address",
    "date",
    "url",
    "account_number",
    "secret",
    "national_id",
    "passport",
    "license_plate",
    "username",
    "nhi_card",
    "company_id",
    "driver_license",
    "medical_license",
    "military_id",
    "household_no",
)
JOIN_KINDS = ("person_part", "address_strong", "address_weak")
SPECIAL_KINDS = (*JOIN_KINDS, "ignore", "outside")
PERSON_GAP = re.compile(r"[\s·・.]{0,2}")
ADDRESS_GAP = re.compile(r"[\s,，、.\-/#]{0,3}")

HAN = re.compile(r"[一-鿿]")
KANA_OR_HANGUL = re.compile(r"[぀-ヿ가-힯]")
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LATIN = re.compile(r"[A-Za-z]")
TRADITIONAL_MARKERS = set("們個來國學電鐵這會說時對過還發麼後見開關條號門間體")
SIMPLIFIED_MARKERS = set("们个来国学电铁这会说时对过还发么后见开关条号门间体")


def script_of(text: str) -> str | None:
    """'latin', 'zh-Hans', 'zh-Hant', 'zh', or None for scripts outside the target."""
    if KANA_OR_HANGUL.search(text) or CYRILLIC.search(text):
        return None
    if HAN.search(text):
        traditional = sum(character in TRADITIONAL_MARKERS for character in text)
        simplified = sum(character in SIMPLIFIED_MARKERS for character in text)
        if traditional == simplified:
            return "zh"
        return "zh-Hant" if traditional > simplified else "zh-Hans"
    return "latin" if LATIN.search(text) else None


def validate_map(label_map: dict[str, str]) -> None:
    unknown = {kind for kind in label_map.values() if kind not in TYPES and kind not in SPECIAL_KINDS}
    if unknown:
        raise ValueError(f"unknown kinds in the label map: {sorted(unknown)}")


def map_spans(spans: list[dict], label_map: dict[str, str]) -> list[dict]:
    """Attach a kind to every span. A label missing from the map is an error, never a silent drop."""
    missing = sorted({span["label"] for span in spans if span["label"] not in label_map})
    if missing:
        raise KeyError(f"labels missing from the map: {missing}")
    return [{"start": s["start"], "end": s["end"], "kind": label_map[s["label"]]} for s in spans]


def join_parts(text: str, spans: list[dict]) -> list[dict]:
    """Join adjacent name parts into person and adjacent address parts into address."""
    joined: list[dict] = []
    for span in sorted(spans, key=lambda item: (item["start"], item["end"])):
        if joined and _joins(text, joined[-1], span):
            joined[-1] = _merge(joined[-1], span)
        else:
            joined.append({**span, "parts": {span["kind"]}})
    return [_resolve(span) for span in joined]


def _joins(text: str, previous: dict, span: dict) -> bool:
    gap = text[previous["end"] : span["start"]]
    if span["start"] < previous["end"]:
        return False
    if previous["kind"] == "person_part" and span["kind"] == "person_part":
        return PERSON_GAP.fullmatch(gap) is not None
    address_kinds = ("address", "address_strong", "address_weak")
    if previous["kind"] in address_kinds and span["kind"] in address_kinds:
        return ADDRESS_GAP.fullmatch(gap) is not None
    return False


def _merge(previous: dict, span: dict) -> dict:
    parts = previous["parts"] | {span["kind"]}
    kind = "person_part" if previous["kind"] == "person_part" else "address_strong"
    if parts <= {"address_weak"}:
        kind = "address_weak"
    return {"start": previous["start"], "end": span["end"], "kind": kind, "parts": parts}


def _resolve(span: dict) -> dict:
    kind = span["kind"]
    if kind == "person_part":
        kind = "person"
    elif kind == "address_strong":
        kind = "address"
    elif kind == "address_weak":
        kind = "outside"
    return {"start": span["start"], "end": span["end"], "kind": kind}


def settle_overlaps(spans: list[dict]) -> tuple[list[dict], list[dict], int]:
    """Split kinds into typed entities and ignore spans, and drop overlapping entities.

    A typed entity beats an ignore span it overlaps. Two overlapping typed entities keep
    the longer one. Returns (entities, ignore spans, number of entities dropped).
    """
    typed = sorted((s for s in spans if s["kind"] in TYPES), key=lambda s: (s["start"], -(s["end"] - s["start"])))
    kept: list[dict] = []
    for span in typed:
        if kept and span["start"] < kept[-1]["end"]:
            kept[-1] = max(kept[-1], span, key=lambda item: item["end"] - item["start"])
            continue
        kept.append(span)
    entities = [{"start": s["start"], "end": s["end"], "label": s["kind"]} for s in kept]
    ignored = [
        {"start": s["start"], "end": s["end"]}
        for s in spans
        if s["kind"] == "ignore" and not any(s["start"] < e["end"] and e["start"] < s["end"] for e in entities)
    ]
    return entities, ignored, len(typed) - len(kept)


def find_all(text: str, needle: str) -> list[tuple[int, int]]:
    """Every non-overlapping occurrence of needle, for sources that give entity strings but no offsets."""
    if not needle:
        return []
    found, start = [], text.find(needle)
    while start != -1:
        found.append((start, start + len(needle)))
        start = text.find(needle, start + len(needle))
    return found


def spans_valid(text: str, spans: list[dict]) -> bool:
    return all(0 <= s["start"] < s["end"] <= len(text) for s in spans)


def is_dev(text: str, share_per_thousand: int = 10) -> bool:
    """Stable hash split, so a row lands on the same side on every build."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 1000 < share_per_thousand
