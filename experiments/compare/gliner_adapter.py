"""NVIDIA gliner-PII as a text-to-spans function in our types.

GLiNER takes the labels to look for from the caller. The model trained on Nemotron-PII's
labels, so it gets those that map to one of our types, with the mapping from type_map.yaml.
Nemotron marks first and last names apart, so adjacent name parts merge into one person.
The card evaluates at threshold 0.3, and so does this adapter.
"""

import re

MODEL_ID = "nvidia/gliner-PII"
THRESHOLD = 0.3
LABELS = {
    "first_name": "person", "last_name": "person",
    "email": "email", "phone_number": "phone", "fax_number": "phone",
    "street_address": "address", "date_of_birth": "date", "url": "url",
    "account_number": "account_number", "credit_debit_card": "account_number",
    "bank_routing_number": "account_number",
    "password": "secret", "pin": "secret", "api_key": "secret", "http_cookie": "secret", "cvv": "secret",
    "ssn": "national_id", "national_id": "national_id",
    "license_plate": "license_plate", "vehicle_identifier": "license_plate", "user_name": "username",
}  # fmt: skip


def our_spans(found: list[dict]) -> list[dict]:
    return [
        {"start": s["start"], "end": s["end"], "label": LABELS[s["label"]]}
        for s in sorted(found, key=lambda s: s["start"])
        if s["label"] in LABELS
    ]


def merged_person_parts(spans: list[dict], text: str) -> list[dict]:
    """Join person spans that only whitespace separates, as a full name is one mark."""
    merged: list[dict] = []
    for span in spans:
        previous = merged[-1] if merged else None
        joinable = previous and previous["label"] == span["label"] == "person"
        if joinable and re.fullmatch(r"\s*", text[previous["end"] : span["start"]]):
            previous["end"] = span["end"]
            continue
        merged.append(dict(span))
    return merged


def predictor(device: str = "mps"):
    import torch
    from gliner import GLiNER

    model = GLiNER.from_pretrained(MODEL_ID)
    model.to(torch.device(device if device != "mps" or torch.backends.mps.is_available() else "cpu"))
    model.eval()

    def predict(text: str) -> list[dict]:
        found = model.predict_entities(text, list(LABELS), threshold=THRESHOLD)
        return merged_person_parts(our_spans(found), text)

    return predict
