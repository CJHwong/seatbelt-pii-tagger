# /// script
# requires-python = ">=3.12"
# dependencies = ["pyyaml", "faker"]
# ///
"""Generate rows where the same digit shape is an ID in one place and a business number in another.

usage: uv run --script scripts/digit_contrast.py [--en N] [--zh-hant N] [--zh-hans N] [--seed S]
    -> data/synthetic/digit_contrast/train.jsonl

The transcript rows teach that a long digit run is not PII: prepare.py scrambles shipment,
order and record numbers in place and leaves them unlabelled, and train3 holds about five
times more of them per row than the earlier transcripts. ID recall on public sets dropped
with them, and an ignore label on them cost recall elsewhere. The guideline already draws
the line: an ID or a bank or card number is PII, an invoice, order or tracking number is
not. These rows show both kinds at the same lengths, so the words around a number decide.
Every value is generated; templates come from the guideline, never from a judged or
benchmark row.
"""

import argparse
import json
import random
import string
from pathlib import Path

import yaml
from faker import Faker

import tw_values as tw
from synth_lib import Slot, check_templates, render

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "synthetic" / "digit_contrast"
LANGUAGES = {"en": "latin", "zh-Hant": "zh-Hant", "zh-Hans": "zh-Hans"}
LABELS = {"national_id", "passport", "driver_license", "account_number", "secret"}
_FAKERS = {locale: Faker(locale) for locale in ("en_US", "zh_CN", "en_GB", "de_DE")}


def _seeded(locale: str, picker: random.Random) -> Faker:
    faker = _FAKERS[locale]
    faker.seed_instance(picker.getrandbits(32))
    return faker


def digits(picker: random.Random, count: int) -> str:
    return "".join(picker.choice(string.digits) for _ in range(count))


def lead_digits(picker: random.Random, count: int) -> str:
    """No leading zero: a business system prints its numbers without one."""
    return picker.choice("123456789") + digits(picker, count - 1)


# IDs, labelled. Shapes follow each issuer's published format.


def ssn(picker: random.Random) -> str:
    value = _seeded("en_US", picker).ssn()  # Faker skips the 000, 666 and 9xx areas
    return value if picker.random() < 0.5 else value.replace("-", "")


def resident_id_cn(picker: random.Random) -> str:
    return _seeded("zh_CN", picker).ssn()  # 18 characters, region code, birth date, check digit


def passport_any(picker: random.Random) -> str:
    """US and UK: 9 digits. Taiwan: 9 digits from 3. Many others: one or two letters, 7 or 8 digits."""
    style = picker.randrange(4)
    if style == 0:
        return lead_digits(picker, 9)
    if style == 1:
        return tw.passport(picker)
    if style == 2:
        return picker.choice("EGK") + digits(picker, 8)  # mainland China
    return "".join(picker.choice(string.ascii_uppercase) for _ in range(picker.randint(1, 2))) + digits(picker, 7)


def driver_license_any(picker: random.Random) -> str:
    """US states: one letter and 7 to 12 digits, or 8 to 9 digits. Taiwan: the national ID."""
    style = picker.randrange(3)
    if style == 0:
        return picker.choice(string.ascii_uppercase) + digits(picker, picker.choice([7, 8, 12]))
    if style == 1:
        return lead_digits(picker, picker.choice([8, 9]))
    return tw.driver_license(picker)


def nino(picker: random.Random) -> str:
    """UK National Insurance number: two letters, six digits, A to D (HMRC manual NIM39110)."""
    letters = "ABCEGHJKLMNPRSTWXYZ"
    body = picker.choice(letters) + picker.choice(letters) + digits(picker, 6) + picker.choice("ABCD")
    return body if picker.random() < 0.6 else " ".join([body[:2], body[2:4], body[4:6], body[6:8], body[8]])


def sin_ca(picker: random.Random) -> str:
    """Canadian Social Insurance Number: 9 digits, often written 3-3-3."""
    value = lead_digits(picker, 9)
    return picker.choice([value, f"{value[:3]} {value[3:6]} {value[6:]}", f"{value[:3]}-{value[3:6]}-{value[6:]}"])


def aadhaar(picker: random.Random) -> str:
    """India Aadhaar: 12 digits, first digit 2 to 9, often written 4-4-4."""
    value = picker.choice("23456789") + digits(picker, 11)
    return value if picker.random() < 0.5 else f"{value[:4]} {value[4:8]} {value[8:]}"


def dni_es(picker: random.Random) -> str:
    """Spanish DNI: 8 digits and the check letter of the number modulo 23."""
    number = picker.randint(10_000_000, 99_999_999)
    return f"{number}{'TRWAGMYFPDXBNJZSQVHLCKE'[number % 23]}"


def bsn_nl(picker: random.Random) -> str:
    """Dutch BSN: 9 digits that pass the eleven test."""
    while True:
        value = lead_digits(picker, 9)
        total = sum(int(digit) * weight for digit, weight in zip(value[:8], range(9, 1, -1))) - int(value[8])
        if total % 11 == 0:
            return value


def uk_licence(picker: random.Random) -> str:
    """UK driving licence: 5 surname letters, 6 digits, 2 initials, a digit, 2 check letters (DVLA)."""
    letter = lambda count: "".join(picker.choice(string.ascii_uppercase) for _ in range(count))
    return letter(5) + digits(picker, 6) + letter(2) + picker.choice(string.digits) + letter(2)


def iban(picker: random.Random) -> str:
    value = _seeded(picker.choice(["en_GB", "de_DE"]), picker).iban()
    return value if picker.random() < 0.5 else " ".join(value[i : i + 4] for i in range(0, len(value), 4))


def bank_account(picker: random.Random) -> str:
    return tw.bank_account(picker)


def card(picker: random.Random) -> str:
    return tw.credit_card(picker)


# Business and system numbers, unlabelled. The lengths overlap the IDs on purpose.


def api_key(picker: random.Random) -> str:
    """A key after a key word, on the line below an account number: the model dropped the
    account when the two shared a window."""
    return "".join(picker.choice(string.ascii_letters + string.digits) for _ in range(32))


def order_no(picker: random.Random) -> str:
    return lead_digits(picker, picker.choice([8, 9, 10, 11, 12]))


def tracking_no(picker: random.Random) -> str:
    return lead_digits(picker, picker.choice([10, 12, 12, 15, 20, 22]))


def epoch(picker: random.Random) -> str:
    seconds = str(picker.randint(1_600_000_000, 1_800_000_000))
    return seconds if picker.random() < 0.6 else seconds + digits(picker, 3)


def run_id(picker: random.Random) -> str:
    return lead_digits(picker, picker.choice([9, 10, 11]))


def size_bytes(picker: random.Random) -> str:
    return lead_digits(picker, picker.choice([6, 7, 8, 9, 10]))


def serial(picker: random.Random) -> str:
    return "".join(picker.choice(string.ascii_uppercase) for _ in range(picker.randint(1, 3))) + digits(
        picker, picker.choice([7, 8, 9, 10])
    )


def pid(picker: random.Random) -> str:
    return str(picker.randint(1_000, 4_194_304))


SLOTS = {
    "ssn": Slot("national_id", ssn),
    "tw_id": Slot("national_id", tw.national_id),
    "cn_id": Slot("national_id", resident_id_cn),
    "passport": Slot("passport", passport_any),
    "driver_license": Slot("driver_license", driver_license_any),
    "nino": Slot("national_id", nino),
    "sin": Slot("national_id", sin_ca),
    "aadhaar": Slot("national_id", aadhaar),
    "dni": Slot("national_id", dni_es),
    "bsn": Slot("national_id", bsn_nl),
    "uk_licence": Slot("driver_license", uk_licence),
    "iban": Slot("account_number", iban),
    "bank_account": Slot("account_number", bank_account),
    "card": Slot("account_number", card),
    "api_key": Slot("secret", api_key),
    "order": Slot(None, order_no),
    "invoice": Slot(None, order_no),
    "tracking": Slot(None, tracking_no),
    "booking": Slot(None, order_no),
    "ticket": Slot(None, order_no),
    "epoch": Slot(None, epoch),
    "run_id": Slot(None, run_id),
    "bytes": Slot(None, size_bytes),
    "pid": Slot(None, pid),
    "customer": Slot(None, order_no),
    "employee": Slot(None, order_no),
    "case": Slot(None, order_no),
    "serial": Slot(None, serial),
}


def load_templates() -> dict[str, list[str]]:
    return yaml.safe_load((ROOT / "synthetic" / "digit_templates.yaml").read_text(encoding="utf-8"))


def generate(language: str, templates: list[str], count: int, picker: random.Random) -> list[dict]:
    """count rows, cycling templates evenly so each template gets the same share."""
    check_templates(templates, SLOTS)
    rows = []
    for index in range(count):
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
    parser.add_argument("--seed", type=int, default=13)
    arguments = parser.parse_args()
    picker = random.Random(arguments.seed)
    templates = load_templates()
    counts = {"en": arguments.en, "zh-Hant": arguments.zh_hant, "zh-Hans": arguments.zh_hans}
    rows = [row for language, count in counts.items() for row in generate(language, templates[language], count, picker)]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "train.jsonl", "w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    print(f"{len(rows)} rows, {sum(not row['spans'] for row in rows)} with no ID")


if __name__ == "__main__":
    main()
