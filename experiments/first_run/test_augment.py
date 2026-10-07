import random

from augment import augmented, lowercased, with_social_marks, without_punctuation


def _texts(row: dict) -> list[str]:
    return [row["text"][s["start"] : s["end"]] for s in row["entities"] + row["ignore"]]


def _row(text: str, parts: list[tuple[str, str]], ignore: list[str] = (), script: str = "zh-Hant") -> dict:
    entities = [{"start": text.index(p), "end": text.index(p) + len(p), "label": label} for p, label in parts]
    ignored = [{"start": text.index(p), "end": text.index(p) + len(p)} for p in ignore]
    return {"text": text, "entities": entities, "ignore": ignored, "script": script}


def test_lowercased_keeps_spans_on_the_same_characters():
    row = _row("Call Ann Lee at ANN@MAIL.COM.", [("Ann Lee", "person"), ("ANN@MAIL.COM", "email")], script="latin")
    out = lowercased(row)
    assert out["text"] == "call ann lee at ann@mail.com."
    assert _texts(out) == ["ann lee", "ann@mail.com"]


def test_lowercased_skips_text_whose_length_changes():
    row = _row("İstanbul Ann", [("Ann", "person")], script="latin")
    assert lowercased(row) is row


def test_without_punctuation_removes_only_outside_spans():
    row = _row(
        "我叫王小明，電話：0912-345-678。地址：台北市。", [("王小明", "person"), ("0912-345-678", "phone")], ["台北市"]
    )
    out = without_punctuation(row)
    assert out["text"] == "我叫王小明電話0912-345-678地址台北市"
    assert _texts(out) == ["王小明", "0912-345-678", "台北市"]


def test_social_marks_insert_outside_spans_and_keep_them_aligned():
    row = _row("我是王小明。今天好累。", [("王小明", "person")])
    out = with_social_marks(row, random.Random(3))
    assert out["text"] != row["text"]
    assert "@" not in out["text"]
    assert _texts(out) == ["王小明"]


def test_augmented_changes_about_the_share_of_chinese_rows_and_never_breaks_a_span():
    picker = random.Random(5)
    rows = [
        _row("Hi, I am Ann Lee. Mail ann@x.com!", [("Ann Lee", "person"), ("ann@x.com", "email")], script="latin"),
        _row("我叫王小明，電話0912345678。", [("王小明", "person"), ("0912345678", "phone")]),
    ] * 200
    out = [augmented(row, picker, share=0.15) for row in rows]
    changed = [new is not old for new, old in zip(out, rows, strict=True)]
    assert not any(changed[0::2])
    assert 10 < sum(changed[1::2]) < 50
    for new, old in zip(out, rows, strict=True):
        assert [t.lower() for t in _texts(new)] == [t.lower() for t in _texts(old)]


def test_social_marks_never_stretch_a_span_that_ends_the_text():
    for seed in range(30):
        row = _row("電話 0912345678", [("0912345678", "phone")])
        out = with_social_marks(row, random.Random(seed))
        assert _texts(out) == ["0912345678"]
