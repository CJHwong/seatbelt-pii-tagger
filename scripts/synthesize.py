# /// script
# requires-python = ">=3.12"
# dependencies = ["pyyaml", "faker"]
# ///
"""Generate Traditional Chinese training rows from synthetic/templates.yaml.

Writes data/synthetic/train.jsonl from the train templates and data/synthetic/heldout.jsonl
from the heldout templates. The two sets use disjoint scenarios, so heldout measures
generalisation to new contexts, not recall of seen templates. heldout never feeds training.

Row format matches what build_training_set.py reads: {"text", "spans": [{start, end, label, value}]}.
The label is already a final type, or "ignore" for a real identifier outside the type list.
"""

import argparse
import json
import random
from pathlib import Path

import yaml

import tw_values as tw
from synth_lib import Slot, check_templates, render

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "synthetic"

SLOTS = {
    # Labelled: the slot name is the Taiwanese field, the label is the type it trains.
    "person": Slot("person", tw.person),
    "email": Slot("email", tw.email),
    "mobile": Slot("phone", tw.mobile),
    "landline": Slot("phone", tw.landline),
    "address": Slot("address", tw.address),
    "dob": Slot("date", tw.dob),
    "personal_url": Slot("url", tw.personal_url),
    "bank_account": Slot("account_number", tw.bank_account),
    "credit_card": Slot("account_number", tw.credit_card),
    "password": Slot("secret", tw.password),
    "api_key": Slot("secret", tw.api_key),
    "otp": Slot("secret", tw.otp),
    "national_id": Slot("national_id", tw.national_id),
    "passport": Slot("passport", tw.passport),
    "license_plate": Slot("license_plate", tw.license_plate),
    "line_id": Slot("username", tw.line_id),
    "ptt_id": Slot("username", tw.ptt_id),
    "ig_handle": Slot("username", tw.ig_handle),
    "nhi_card": Slot("nhi_card", tw.nhi_card),
    "company_id": Slot("company_id", tw.company_id),
    "driver_license": Slot("driver_license", tw.driver_license),
    "medical_license": Slot("medical_license", tw.medical_license),
    "military_id": Slot("military_id", tw.military_id),
    "household_no": Slot("household_no", tw.household_no),
    # A hospital record number identifies a patient but has no type yet, so the loss skips it.
    "patient_id": Slot("ignore", lambda picker: picker.choice("ABCDEFGH") + tw.digits(7)(picker)),
    # Public: written as plain text, so the model learns these are not PII.
    "store": Slot(None, tw.public_choice(tw.STORES)),
    "landmark": Slot(None, tw.public_choice(tw.LANDMARKS)),
    "landmark_address": Slot(None, tw.public_choice(tw.LANDMARK_ADDRESSES)),
    "hotline": Slot(None, tw.public_choice(tw.HOTLINES)),
    "org_email": Slot(None, tw.public_choice(tw.ORG_EMAILS)),
    "celebrity": Slot(None, tw.public_choice(tw.CELEBRITIES)),
    "public_url": Slot(None, tw.public_choice(tw.PUBLIC_URLS)),
    "product": Slot(None, tw.public_choice(tw.PRODUCTS)),
    "hospital": Slot(None, tw.public_choice(tw.HOSPITALS)),
    "school": Slot(None, tw.public_choice(tw.SCHOOLS)),
    "other_date": Slot(None, tw.other_date),
    "city": Slot(None, tw.city),
    "amount": Slot(None, tw.amount),
    "time": Slot(None, tw.time_of_day),
    "bank_name": Slot(None, tw.bank_name),
    "bank_code": Slot(None, tw.bank_code),
    "surname": Slot(None, tw.surname),
    "card_tail": Slot(None, tw.digits(4)),
    "phone_tail": Slot(None, tw.digits(3)),
    "extension": Slot(None, tw.digits(3)),
    "table_no": Slot(None, lambda picker: str(picker.randint(1, 30))),
}


def generate(templates: list[str], count: int, picker: random.Random) -> list[dict]:
    """count rows, cycling templates evenly so every template gets the same share."""
    rows = []
    for index in range(count):
        row = render(templates[index % len(templates)], SLOTS, picker)
        spans = [{**entity, "value": row["text"][entity["start"] : entity["end"]]} for entity in row["entities"]]
        rows.append({"text": row["text"], "spans": spans})
    picker.shuffle(rows)
    return rows


def write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=50_000)
    parser.add_argument("--heldout", type=int, default=3_000)
    parser.add_argument("--seed", type=int, default=13)
    arguments = parser.parse_args()
    templates = yaml.safe_load((ROOT / "synthetic" / "templates.yaml").read_text(encoding="utf-8"))
    for split in ("train", "heldout"):
        check_templates(templates[split], SLOTS)
    OUT.mkdir(parents=True, exist_ok=True)
    picker = random.Random(arguments.seed)
    for split, count in (("train", arguments.train), ("heldout", arguments.heldout)):
        rows = generate(templates[split], count, picker)
        write(OUT / f"{split}.jsonl", rows)
        print(f"{split}: {len(rows)} rows from {len(templates[split])} templates")


if __name__ == "__main__":
    main()
