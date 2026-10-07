from compare_lib import category, mark_leaked, normalized, outside, row_counts, summary, text_key, unleaked


def test_hook_labels_map_into_our_names():
    assert normalized("private_person") == "person"
    assert normalized("national_id") == "national_id"


def test_ids_fold_into_account_number_and_unshared_types_drop():
    assert category("nhi_card") == "account_number"
    assert category("username") is None
    assert category("date") is None


def test_a_lead_in_word_is_a_typed_overlap_hit_but_not_strict():
    gold = [{"start": 3, "end": 13, "label": "national_id"}]
    predicted = [{"start": 0, "end": 13, "label": "account_number"}]
    report = summary(row_counts(gold, predicted))
    assert report["typed_strict"]["f1"] == 0.0
    assert report["typed_overlap"]["f1"] == 1.0


def test_dates_count_on_neither_side():
    gold = [{"start": 0, "end": 5, "label": "date"}]
    predicted = [{"start": 0, "end": 5, "label": "date"}, {"start": 10, "end": 12, "label": "date"}]
    counts = row_counts(gold, predicted)
    assert counts["any_gold"] == 0 and counts["any_predicted"] == 0


def test_wrong_category_counts_for_untyped_only():
    gold = [{"start": 0, "end": 3, "label": "person"}]
    predicted = [{"start": 0, "end": 3, "label": "address"}]
    report = summary(row_counts(gold, predicted))
    assert report["typed_overlap"]["recall"] == 0.0
    assert report["any_overlap"]["recall"] == 1.0


def test_outside_drops_spans_on_a_dont_care_region():
    spans = [{"start": 0, "end": 3, "label": "phone"}, {"start": 5, "end": 8, "label": "phone"}]
    assert outside(spans, [{"start": 2, "end": 4}]) == [spans[1]]


def test_text_key_ignores_whitespace():
    assert text_key("Ann  Lee\nwrote") == text_key("Ann Lee wrote")
    assert text_key("Ann Lee wrote") != text_key("Ann Lee read")


def test_mark_leaked_marks_only_the_listed_texts_of_that_set():
    sets = {"a": [{"text": "one"}, {"text": "two"}], "b": [{"text": "one"}]}
    marked = mark_leaked(sets, {"a": [text_key("one")]})
    assert [row.get("leaked", False) for row in marked["a"]] == [True, False]
    assert [row.get("leaked", False) for row in marked["b"]] == [False]
    assert "leaked" not in sets["a"][0]


def test_unleaked_drops_marked_rows_and_reindexes_their_predictions():
    rows = [{"text": "x", "leaked": True}, {"text": "y"}, {"text": "z"}]
    predictions = {0: ["p0"], 2: ["p2"]}
    kept, kept_predictions = unleaked(rows, predictions)
    assert [row["text"] for row in kept] == ["y", "z"]
    assert kept_predictions == {1: ["p2"]}
