import pytest

from prep_lib import (
    find_all,
    is_dev,
    join_parts,
    map_spans,
    script_of,
    settle_overlaps,
    spans_valid,
    validate_map,
)


def span(start, end, kind):
    return {"start": start, "end": end, "kind": kind}


def test_script_of_separates_the_targets():
    assert script_of("my email is a@b.co") == "latin"
    assert script_of("我们的电话") == "zh-Hans"
    assert script_of("我們的電話") == "zh-Hant"
    assert script_of("張三") == "zh"
    assert script_of("こんにちは東京") is None
    assert script_of("Привет") is None


def test_validate_map_rejects_an_unknown_kind():
    with pytest.raises(ValueError):
        validate_map({"NAME": "persn"})


def test_map_spans_refuses_a_missing_label():
    with pytest.raises(KeyError):
        map_spans([{"start": 0, "end": 1, "label": "NEW"}], {"OLD": "person"})


def test_map_spans_attaches_kinds():
    assert map_spans([{"start": 0, "end": 3, "label": "EMAIL"}], {"EMAIL": "email"}) == [span(0, 3, "email")]


def test_name_parts_join_across_a_space_and_with_no_gap():
    text = "Ann Lee 和 张伟"
    spans = [
        span(0, 3, "person_part"),
        span(4, 7, "person_part"),
        span(10, 11, "person_part"),
        span(11, 12, "person_part"),
    ]
    assert join_parts(text, spans) == [span(0, 7, "person"), span(10, 12, "person")]


def test_names_split_by_a_comma_stay_apart():
    text = "Ann, Lee"
    assert join_parts(text, [span(0, 3, "person_part"), span(5, 8, "person_part")]) == [
        span(0, 3, "person"),
        span(5, 8, "person"),
    ]


def test_address_joins_strong_and_weak_parts():
    text = "12 Main St, Springfield"
    spans = [span(0, 10, "address_strong"), span(12, 23, "address_weak")]
    assert join_parts(text, spans) == [span(0, 23, "address")]


def test_a_lone_city_is_outside():
    assert join_parts("in Paris today", [span(3, 8, "address_weak")]) == [span(3, 8, "outside")]


def test_weak_parts_joined_together_stay_outside():
    text = "Paris, France"
    assert join_parts(text, [span(0, 5, "address_weak"), span(7, 13, "address_weak")]) == [span(0, 13, "outside")]


def test_a_full_address_absorbs_an_adjacent_city():
    text = "北京市朝阳区人民路224号"
    spans = [span(0, 3, "address_weak"), span(3, 13, "address")]
    assert join_parts(text, spans) == [span(0, 13, "address")]


def test_words_between_parts_block_the_join():
    text = "12 Main St, city of Springfield"
    spans = [span(0, 10, "address_strong"), span(20, 31, "address_weak")]
    assert join_parts(text, spans) == [span(0, 10, "address"), span(20, 31, "outside")]


def test_other_kinds_pass_through():
    assert join_parts("a@b.co", [span(0, 6, "email")]) == [span(0, 6, "email")]


def test_settle_overlaps_prefers_typed_and_longer():
    spans = [
        span(0, 10, "address"),
        span(3, 6, "person"),
        span(12, 15, "ignore"),
        span(4, 8, "ignore"),
        span(20, 24, "outside"),
    ]
    entities, ignored, dropped = settle_overlaps(spans)
    assert entities == [{"start": 0, "end": 10, "label": "address"}]
    assert ignored == [{"start": 12, "end": 15}]
    assert dropped == 1


def test_find_all_returns_every_occurrence():
    assert find_all("Ann met Ann", "Ann") == [(0, 3), (8, 11)]
    assert find_all("x", "") == []


def test_spans_valid_checks_bounds():
    assert spans_valid("abc", [{"start": 0, "end": 3}])
    assert not spans_valid("abc", [{"start": 2, "end": 4}])
    assert not spans_valid("abc", [{"start": 2, "end": 2}])


def test_is_dev_is_stable_and_near_its_share():
    rows = [f"row {index}" for index in range(20_000)]
    share = sum(is_dev(row) for row in rows) / len(rows)
    assert is_dev("row 1") == is_dev("row 1")
    assert 0.007 < share < 0.013
