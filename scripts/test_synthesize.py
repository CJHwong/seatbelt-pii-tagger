import random
from pathlib import Path

import yaml

from prep_lib import TYPES
from synth_lib import check_templates, slot_names
from synthesize import SLOTS, generate

TEMPLATES = yaml.safe_load((Path(__file__).resolve().parent.parent / "synthetic/templates.yaml").read_text())


def test_every_slot_label_is_a_type_ignore_or_public():
    assert {slot.label for slot in SLOTS.values()} <= {*TYPES, "ignore", None}


def test_all_templates_use_only_registered_slots():
    check_templates(TEMPLATES["train"] + TEMPLATES["heldout"], SLOTS)


def test_train_and_heldout_share_no_template():
    assert not set(TEMPLATES["train"]) & set(TEMPLATES["heldout"])


def test_generated_spans_point_at_their_values():
    rows = generate(TEMPLATES["train"] + TEMPLATES["heldout"], 2_000, random.Random(0))
    for row in rows:
        for span in row["spans"]:
            assert row["text"][span["start"] : span["end"]] == span["value"]
            assert span["value"].strip() == span["value"]


def test_every_labelled_slot_is_used_by_some_template():
    used = set().union(*(slot_names(template) for template in TEMPLATES["train"] + TEMPLATES["heldout"]))
    labelled = {name for name, slot in SLOTS.items() if slot.label}
    assert labelled <= used
