# /// script
# requires-python = ">=3.12"
# dependencies = ["pyyaml", "faker"]
# ///
"""Generate rows for the false alarms the release raised in real sessions, and for the names it missed.

usage: uv run --script scripts/field_contrast.py [--en N] [--zh-hant N] [--zh-hans N] [--seed S]
    -> data/synthetic/field_contrast/train.jsonl

Field false alarms (locale cookies, mixed-case code identifiers, locale tags, numbers keyed
by a PII label name) sit next to the real PII of the same frame: session cookies, names and
handles in git and Slack output. Judged dev misses most lone names and handles inside tool
output, so those frames carry labelled names too. Every value is generated.
"""

import argparse
import json
import random
import string
from pathlib import Path

import yaml
from faker import Faker

from synth_lib import Slot, check_templates, render

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "synthetic" / "field_contrast"
LANGUAGES = {"en": "latin", "zh-Hant": "zh-Hant", "zh-Hans": "zh-Hans"}
LABELS = {"person", "username", "secret"}
_FAKERS = {locale: Faker(locale) for locale in ("en_US", "zh_TW", "zh_CN")}
SERVICES = [
    "postgres",
    "redis-cache",
    "api-gateway",
    "worker",
    "scheduler",
    "billing-svc",
    "search",
    "nginx",
    "kafka",
    "metrics",
]


def _seeded(locale: str, picker: random.Random) -> Faker:
    faker = _FAKERS[locale]
    faker.seed_instance(picker.getrandbits(32))
    return faker


def lang_cookie(picker: random.Random) -> str:
    name = picker.choice(["django_language", "locale", "lang", "i18n_redirected", "NEXT_LOCALE"])
    return f"{name}={picker.choice(['en', 'en-gb', 'zh-hant', 'zh-tw', 'zh-hans', 'ja', 'de', 'fr'])}"


def locale(picker: random.Random) -> str:
    return picker.choice(["zh-Hant", "zh-Hans", "zh-TW", "zh-CN", "en-US", "en-GB", "ja-JP", "pt-BR", "fr-FR"])


def code_id(picker: random.Random) -> str:
    """Capitals, an underscore, then mixed case and digits, as generated system field names."""
    head = "".join(picker.choice(string.ascii_uppercase) for _ in range(picker.randint(3, 5)))
    return (
        head + "_" + "".join(picker.choice(string.ascii_letters + string.digits) for _ in range(picker.randint(10, 18)))
    )


def metric(picker: random.Random) -> str:
    return f"{picker.random():.{picker.choice([3, 4, 16])}f}"


def count(picker: random.Random) -> str:
    return str(picker.randint(2, 999))


def sha(picker: random.Random) -> str:
    return "".join(picker.choice("0123456789abcdef") for _ in range(picker.choice([7, 11, 40])))


def service(picker: random.Random) -> str:
    return picker.choice(SERVICES)


def session_token(picker: random.Random) -> str:
    return "".join(picker.choice(string.ascii_lowercase + string.digits) for _ in range(32))


def first(picker: random.Random) -> str:
    faker = _seeded("en_US", picker)
    return faker.first_name() if picker.random() < 0.7 else faker.last_name()


def full(picker: random.Random) -> str:
    return _seeded(picker.choice(["zh_TW", "zh_CN"]), picker).name()


def handle(picker: random.Random) -> str:
    return _seeded("en_US", picker).user_name()


SLOTS = {
    "lang_cookie": Slot(None, lang_cookie),
    "locale": Slot(None, locale),
    "code_id": Slot(None, code_id),
    "metric": Slot(None, metric),
    "count": Slot(None, count),
    "sha": Slot(None, sha),
    "service": Slot(None, service),
    "session_token": Slot("secret", session_token),
    "first": Slot("person", first),
    "full": Slot("person", full),
    "handle": Slot("username", handle),
}


def load_templates() -> dict[str, list[str]]:
    return yaml.safe_load((ROOT / "synthetic" / "field_templates.yaml").read_text(encoding="utf-8"))


def generate(language: str, templates: list[str], total: int, picker: random.Random) -> list[dict]:
    """total rows, cycling templates evenly so each template gets the same share."""
    check_templates(templates, SLOTS)
    rows = []
    for index in range(total):
        row = render(templates[index % len(templates)], SLOTS, picker)
        spans = [{**entity, "value": row["text"][entity["start"] : entity["end"]]} for entity in row["entities"]]
        rows.append({"text": row["text"], "spans": spans, "script": LANGUAGES[language]})
    picker.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--en", type=int, default=6_000)
    parser.add_argument("--zh-hant", type=int, default=2_000)
    parser.add_argument("--zh-hans", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=17)
    arguments = parser.parse_args()
    picker = random.Random(arguments.seed)
    templates = load_templates()
    totals = {"en": arguments.en, "zh-Hant": arguments.zh_hant, "zh-Hans": arguments.zh_hans}
    rows = [row for language, total in totals.items() for row in generate(language, templates[language], total, picker)]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "train.jsonl", "w", encoding="utf-8") as handle_out:
        handle_out.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    print(f"{len(rows)} rows, {sum(not row['spans'] for row in rows)} with no label")


if __name__ == "__main__":
    main()
