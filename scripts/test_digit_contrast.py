import random
import re
from collections import defaultdict

from digit_contrast import LABELS, LANGUAGES, generate, load_templates

DIGIT_RUN = re.compile(r"\d{5,}")
ROWS = 1_500


def test_every_template_renders_in_every_language():
    templates = load_templates()
    assert set(templates) == set(LANGUAGES)
    for language, items in templates.items():
        rows = generate(language, items, len(items), random.Random(1))
        assert len(rows) == len(items)


def test_spans_slice_to_their_values_with_known_labels():
    for language, items in load_templates().items():
        for row in generate(language, items, ROWS, random.Random(2)):
            assert row["script"] == LANGUAGES[language]
            for span in row["spans"]:
                assert span["label"] in LABELS
                assert row["text"][span["start"] : span["end"]] == span["value"]


def test_template_text_carries_no_digit_run_of_its_own():
    """Every long digit run comes from a slot, so the slot alone decides its label."""
    for items in load_templates().values():
        for template in items:
            assert not DIGIT_RUN.search(template), template[:60]


def test_labelled_lengths_also_occur_unlabelled():
    """The contrast: a digit length the model sees as an ID it also sees as a business number."""
    for language, items in load_templates().items():
        lengths = defaultdict(set)
        for row in generate(language, items, ROWS, random.Random(3)):
            covered = [(span["start"], span["end"]) for span in row["spans"]]
            for match in DIGIT_RUN.finditer(row["text"]):
                inside = any(start <= match.start() and match.end() <= end for start, end in covered)
                lengths[inside].add(len(match.group(0)))
        shared = {length for length in lengths[True] if 8 <= length <= 12} & lengths[False]
        assert len(shared) >= 3, (language, sorted(lengths[True]), sorted(lengths[False]))


def test_rows_with_ids_without_and_mixed_all_occur():
    for language, items in load_templates().items():
        rows = generate(language, items, ROWS, random.Random(4))
        with_id = [row for row in rows if row["spans"]]
        mixed = [row for row in with_id if len(DIGIT_RUN.findall(row["text"])) > len(row["spans"])]
        assert 0.3 * len(rows) < len(with_id) < 0.8 * len(rows), language
        assert len(mixed) > 0.2 * len(rows), language


def test_an_account_and_a_key_share_some_windows():
    """The model dropped an account number when a key line followed it in one window."""
    for language, items in load_templates().items():
        rows = generate(language, items, ROWS, random.Random(5))
        both = [row for row in rows if {"account_number", "secret"} <= {span["label"] for span in row["spans"]}]
        assert both, language
