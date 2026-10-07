import random

import pytest

from synth_lib import Slot, check_templates, render, resolve_variants, slot_names

SLOTS = {
    "person": Slot("person", lambda picker: picker.choice(["王小明", "陳美惠"])),
    "phone": Slot("phone", lambda picker: "0912-345-678"),
    "store": Slot(None, lambda picker: "全家便利商店"),
}


def test_render_places_every_labelled_span_exactly():
    picker = random.Random(1)
    for _ in range(50):
        row = render("[您好|哈囉]，我是{person}，電話{phone}，約在{store}見。", SLOTS, picker)
        labels = [entity["label"] for entity in row["entities"]]
        assert labels == ["person", "phone"]
        for entity in row["entities"]:
            value = row["text"][entity["start"] : entity["end"]]
            assert value in ("王小明", "陳美惠", "0912-345-678")


def test_public_slots_write_text_without_a_span():
    row = render("在{store}", SLOTS, random.Random(0))
    assert row == {"text": "在全家便利商店", "entities": []}


def test_a_variant_may_hold_a_slot():
    row = render("[{phone}|{phone}]", SLOTS, random.Random(0))
    assert row["entities"] == [{"start": 0, "end": 12, "label": "phone"}]


def test_empty_variant_is_allowed():
    texts = {resolve_variants("a[b|]c", random.Random(seed)) for seed in range(30)}
    assert texts == {"abc", "ac"}


def test_slot_names_sees_slots_inside_variants():
    assert slot_names("{person}[{phone}|x]") == {"person", "phone"}


def test_check_templates_rejects_unknown_slots_and_nesting():
    with pytest.raises(KeyError):
        check_templates(["{nobody}"], SLOTS)
    with pytest.raises(ValueError):
        check_templates(["[a|[b|c]]"], SLOTS)
