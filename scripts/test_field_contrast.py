import json
import random
from pathlib import Path

from field_contrast import LABELS, LANGUAGES, generate, load_templates

ROWS = 1_200
FIELD_SET = Path(__file__).resolve().parents[2] / "scratch-research" / "field_fp.jsonl"


def test_every_template_renders_in_every_language():
    templates = load_templates()
    assert set(templates) == set(LANGUAGES)
    for language, items in templates.items():
        assert len(generate(language, items, len(items), random.Random(1))) == len(items)


def test_spans_slice_to_their_values_with_known_labels():
    for language, items in load_templates().items():
        for row in generate(language, items, ROWS, random.Random(2)):
            for span in row["spans"]:
                assert span["label"] in LABELS
                assert row["text"][span["start"] : span["end"]] == span["value"]


def test_labelled_and_unlabelled_rows_both_occur():
    for language, items in load_templates().items():
        rows = generate(language, items, ROWS, random.Random(3))
        labelled = sum(bool(row["spans"]) for row in rows)
        assert 0.2 * len(rows) < labelled < 0.8 * len(rows), language


def test_no_row_repeats_a_field_evaluation_line():
    """The field set stays an evaluation set: no generated row may contain one of its lines."""
    if not FIELD_SET.exists():
        return
    lines = {
        line
        for row in map(json.loads, FIELD_SET.open(encoding="utf-8"))
        for line in row["text"].splitlines()
        if len(line) > 20
    }
    for language, items in load_templates().items():
        for row in generate(language, items, ROWS, random.Random(4)):
            assert not any(line in row["text"] for line in lines)
