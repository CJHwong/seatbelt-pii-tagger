import pytest

from build import parse_marked


def test_marks_become_exact_offsets():
    text, spans = parse_marked("欸[[person:陳怡君]]的電話[[phone:0912 345 678]]")
    assert text == "欸陳怡君的電話0912 345 678"
    assert spans == [
        {"start": 1, "end": 4, "label": "person"},
        {"start": 7, "end": 19, "label": "phone"},
    ]


def test_a_value_may_hold_a_colon():
    text, spans = parse_marked("[[url:https://a.b/c]]")
    assert text == "https://a.b/c"
    assert spans == [{"start": 0, "end": 13, "label": "url"}]


def test_unmarked_text_has_no_spans():
    assert parse_marked("全家便利商店") == ("全家便利商店", [])


@pytest.mark.parametrize("bad", ["[[person:王]", "[[nobody:王]]", "[[person:]]", "王]]", "[[person:[[x]]"])
def test_malformed_marks_are_rejected(bad):
    with pytest.raises(ValueError):
        parse_marked(bad)
