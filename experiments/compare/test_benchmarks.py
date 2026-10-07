from benchmarks import argilla_row, case_label, filled_spy_row, score_cases
from gliner_adapter import merged_person_parts, our_spans


def test_spy_placeholders_become_values_with_spans_on_them():
    tokens = ["I ", "am ", "<NAME>", ", ", "mail ", "<EMAIL> ", "or ", "<URL>", "."]
    tags = ["O", "O", "B-NAME", "O", "O", "B-EMAIL", "O", "B-URL", "O"]
    profile = {"NAME": "Ann Lee", "EMAIL": "ann@example.org", "URL": "https://shop.example.com/"}
    row = filled_spy_row(tokens, tags, profile)
    assert row["text"] == "I am Ann Lee, mail ann@example.org or https://shop.example.com/."
    assert [(row["text"][e["start"] : e["end"]], e["label"]) for e in row["entities"]] == [
        ("Ann Lee", "person"),
        ("ann@example.org", "email"),
    ]
    assert [row["text"][d["start"] : d["end"]] for d in row["dont_care"]] == ["https://shop.example.com/"]


def test_spy_id_number_takes_the_type_of_its_generator():
    row = filled_spy_row(["ID ", "<ID_NUM>"], ["O", "B-ID_NUM"], {"ID_NUM": ("123-45-6789", "national_id")})
    assert row["entities"] == [{"start": 3, "end": 14, "label": "national_id"}]


def test_argilla_name_parts_merge_and_a_lone_city_is_ignored():
    text = "Mr. Ann Lee lives at 12 Oak Street, Springfield. Visit Denver soon."
    spans = [
        {"start": 0, "end": 3, "label": "PREFIX"},
        {"start": 4, "end": 7, "label": "FIRSTNAME"},
        {"start": 8, "end": 11, "label": "LASTNAME"},
        {"start": 21, "end": 23, "label": "BUILDINGNUMBER"},
        {"start": 24, "end": 34, "label": "STREET"},
        {"start": 36, "end": 47, "label": "CITY"},
        {"start": 55, "end": 61, "label": "CITY"},
    ]
    row = argilla_row(text, spans)
    assert [(text[e["start"] : e["end"]], e["label"]) for e in row["entities"]] == [
        ("Ann Lee", "person"),
        ("12 Oak Street, Springfield", "address"),
    ]
    assert [text[d["start"] : d["end"]] for d in row["dont_care"]] == ["Denver"]


def test_case_labels_fold_ids_into_account_number():
    assert case_label("private_person") == "person"
    assert case_label("national_id") == "account_number"
    assert case_label("secret") == "secret"


def test_cases_are_scored_like_the_suite_harness():
    cases = [
        {"class": "positive", "expected": ["secret"]},
        {"class": "positive", "expected": ["secret"]},
        {"class": "clean", "expected": []},
        {"class": "ambiguous", "expected": []},
        {"class": "robustness", "expected": ["secret"]},
    ]
    predictions = [[{"label": "secret"}], [{"label": "person"}], [{"label": "date"}], [], [{"label": "secret"}]]
    report = score_cases(cases, predictions)
    assert report["positive"] == {"passed": 1, "total": 2}
    assert report["clean"] == {"flagged": 1, "total": 1}
    assert report["ambiguous"] == {"flagged": 0, "total": 1}
    assert report["robustness"] == {"passed": 1, "total": 1}


def test_gliner_labels_map_and_adjacent_name_parts_merge():
    text = "Ann Lee, ann@example.org"
    found = [
        {"start": 0, "end": 3, "label": "first_name"},
        {"start": 4, "end": 7, "label": "last_name"},
        {"start": 9, "end": 24, "label": "email"},
        {"start": 9, "end": 12, "label": "city"},
    ]
    assert merged_person_parts(our_spans(found), text) == [
        {"start": 0, "end": 7, "label": "person"},
        {"start": 9, "end": 24, "label": "email"},
    ]
