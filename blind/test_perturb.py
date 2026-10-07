from perturb import fullwidth, joined, ocr, rewrite, spaced


def row(text, entities, ignore=()):
    return {"id": "X", "text": text, "entities": list(entities), "ignore": list(ignore)}


def values(result):
    return [(result["text"][e["start"] : e["end"]], e["label"]) for e in result["entities"]]


def test_fullwidth_changes_digits_everywhere_and_keeps_offsets():
    original = row("call 0912345678 at 9", [{"start": 5, "end": 15, "label": "phone"}])
    result = rewrite(original, fullwidth, fullwidth)
    assert values(result) == [("０９１２３４５６７８", "phone")]
    assert result["text"].endswith("９")


def test_spaced_splits_long_digit_runs_inside_numeric_spans_only():
    original = row("ID A123456789, room 123456", [{"start": 3, "end": 13, "label": "national_id"}])
    result = rewrite(original, spaced, lambda text, _label: text)
    assert values(result) == [("A1234 5678 9", "national_id")]
    assert result["text"].endswith("room 123456")


def test_spaced_leaves_a_person_alone():
    original = row("Ann 12345678", [{"start": 0, "end": 3, "label": "person"}])
    assert rewrite(original, spaced, lambda text, _label: text) == original


def test_ocr_swaps_lookalike_characters_inside_spans_and_is_repeatable():
    original = row("x 0101010101 y", [{"start": 2, "end": 12, "label": "phone"}])
    first, second = (
        rewrite(original, ocr, lambda text, _label: text),
        rewrite(original, ocr, lambda text, _label: text),
    )
    assert first == second
    assert first["text"] != original["text"]
    assert set(values(first)[0][0]) <= set("01Ol")


def test_ignore_spans_move_with_the_text():
    original = row("a 1234567 b 7654321", [{"start": 2, "end": 9, "label": "phone"}], [{"start": 12, "end": 19}])
    result = rewrite(original, spaced, spaced)
    assert result["text"][result["ignore"][0]["start"] : result["ignore"][0]["end"]] == "7654 321"


def test_joined_rows_keep_every_span_on_its_value():
    first = row("Ann here", [{"start": 0, "end": 3, "label": "person"}])
    second = row("call 0912", [{"start": 5, "end": 9, "label": "phone"}])
    result = joined([first, second], "\n\n")
    assert values(result) == [("Ann", "person"), ("0912", "phone")]
