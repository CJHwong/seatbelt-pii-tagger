import random

from augment import PLAIN_SWAP_WORDS, TOOL_SWAP_WORDS, augmented, context_swapped, lone_named, tool_lined, lowercased, with_lone_name, with_social_marks, without_punctuation


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


def test_lone_name_adds_one_single_token_span_and_keeps_the_others():
    row = _row("cd /repo\nAsk Ann Lee about it\nmake test", [("Ann Lee", "person")], script="latin")
    for seed in range(50):
        out = with_lone_name(row, random.Random(seed))
        added = [s for s in out["entities"] if out["text"][s["start"] : s["end"]] != "Ann Lee"]
        assert _texts(out).count("Ann Lee") == 1
        assert len(added) == 1 and " " not in out["text"][added[0]["start"] : added[0]["end"]]
        assert out["text"].count("\n") == row["text"].count("\n") + 1


def test_lone_name_never_splits_a_span_across_a_line():
    row = _row("Ann\nLee", [("Ann\nLee", "person")], script="latin")
    for seed in range(20):
        out = with_lone_name(row, random.Random(seed))
        assert "Ann\nLee" in _texts(out)


def test_lone_named_touches_only_developer_sources():
    row = {**_row("Ask Ann", [("Ann", "person")], script="latin"), "source": "nvidia/Nemotron-PII"}
    assert lone_named(row, random.Random(1), 1.0) is row
    assert lone_named({**row, "source": "synthetic/llm-devwork"}, random.Random(1), 1.0) is not row


def test_plain_swap_never_puts_a_tool_name_in_a_person_span():
    row = _row("Ask Ann Lee or Bo about it.", [("Ann Lee", "person"), ("Bo", "person")], script="latin")
    for seed in range(50):
        out = context_swapped(row, random.Random(seed), PLAIN_SWAP_WORDS)
        words = {w for text in _texts(out) for w in text.split()}
        assert words <= set(PLAIN_SWAP_WORDS) and not words & set(TOOL_SWAP_WORDS)


def test_tool_line_adds_text_but_no_span():
    row = {**_row("cd /repo\nAsk Ann Lee\n", [("Ann Lee", "person")], script="latin"), "source": "synthetic/llm-agentwork"}
    for seed in range(30):
        out = tool_lined(row, random.Random(seed), 1.0)
        assert _texts(out) == ["Ann Lee"]
        assert out["text"].count("\n") == row["text"].count("\n") + 1
