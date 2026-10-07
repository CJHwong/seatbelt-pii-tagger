"""Render synthetic training rows from templates, placing spans as the text is built.

Template syntax:
  [a|b|c]   pick one variant. Variants may be empty, [a|], and must not nest.
  {slot}    fill from a registered slot. A labelled slot records a span; a public slot
            (label None) writes text only, which trains the model that it is not PII.
Variants are resolved first, then slots, so a variant may contain a slot.
"""

import random
import re
from collections.abc import Callable
from dataclasses import dataclass

VARIANT = re.compile(r"\[([^\[\]]*)\]")
SLOT = re.compile(r"\{(\w+)\}")


@dataclass(frozen=True)
class Slot:
    label: str | None
    fill: Callable[[random.Random], str]


def resolve_variants(template: str, picker: random.Random) -> str:
    return VARIANT.sub(lambda match: picker.choice(match.group(1).split("|")), template)


def slot_names(template: str) -> set[str]:
    return set(SLOT.findall(VARIANT.sub(lambda match: " ".join(match.group(1).split("|")), template)))


def render(template: str, slots: dict[str, Slot], picker: random.Random) -> dict:
    """One row: {"text", "entities"} with exact character offsets for every labelled slot."""
    text = resolve_variants(template, picker)
    pieces, entities, cursor, length = [], [], 0, 0
    for match in SLOT.finditer(text):
        literal = text[cursor : match.start()]
        pieces.append(literal)
        length += len(literal)
        slot = slots[match.group(1)]
        value = slot.fill(picker)
        if slot.label:
            entities.append({"start": length, "end": length + len(value), "label": slot.label})
        pieces.append(value)
        length += len(value)
        cursor = match.end()
    pieces.append(text[cursor:])
    return {"text": "".join(pieces), "entities": entities}


def check_templates(templates: list[str], slots: dict[str, Slot]) -> None:
    """Fail loudly on an unknown slot or a nested variant before generating anything."""
    for template in templates:
        unknown = slot_names(template) - set(slots)
        if unknown:
            raise KeyError(f"unknown slots {sorted(unknown)} in: {template[:60]}")
        if re.search(r"\[[^\]]*\[", template):
            raise ValueError(f"nested variants in: {template[:60]}")
