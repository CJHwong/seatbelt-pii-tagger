from external import bio_spans, credential_row, spans_of_values


def test_every_occurrence_is_marked_and_the_longer_value_wins():
    text = "Ann Lee called Ann. Lee Street"
    spans = spans_of_values(text, [("Ann Lee", "person"), ("Ann", "person"), ("Lee Street", "address")])
    assert [(text[s["start"] : s["end"]], s["label"]) for s in spans] == [
        ("Ann Lee", "person"),
        ("Ann", "person"),
        ("Lee Street", "address"),
    ]


def test_a_value_inside_a_word_is_not_marked():
    assert spans_of_values("Annabel", [("Ann", "person")]) == []


def test_bio_tags_become_character_spans_after_joining():
    tokens, tags = (["I", "met", "Ann", "Lee", "today"], ["O", "O", "B-person", "I-person", "O"])
    text, spans = bio_spans(tokens, tags, " ", {"person": "person"})
    assert text == "I met Ann Lee today"
    assert [text[s["start"] : s["end"]] for s in spans] == ["Ann Lee"]


def test_an_unmapped_bio_type_is_dropped_and_chinese_joins_without_spaces():
    text, spans = bio_spans(
        ["王", "小", "明", "在", "北", "京"],
        ["B-PER.NAM", "I-PER.NAM", "I-PER.NAM", "O", "B-GPE.NAM", "I-GPE.NAM"],
        "",
        {"PER.NAM": "person"},
    )
    assert text == "王小明在北京"
    assert spans == [{"start": 0, "end": 3, "label": "person"}]


def test_a_credential_value_offset_counts_from_its_own_line():
    lines = ["x = 1", 'password = "hunter22"', "y = 2"]
    row = credential_row(lines, line_start=2, line_end=2, value_start=12, value_end=20)
    assert row["text"] == 'password = "hunter22"'
    assert row["text"][row["entities"][0]["start"] : row["entities"][0]["end"]] == "hunter22"


def test_a_multi_line_key_spans_from_the_first_line_to_the_last():
    lines = ["-----BEGIN KEY-----", "AAAA", "BBBB", "-----END KEY-----"]
    row = credential_row(lines, line_start=1, line_end=4, value_start=0, value_end=17)
    assert row["text"][row["entities"][0]["start"] : row["entities"][0]["end"]] == "\n".join(lines)


def test_a_false_line_has_no_gold():
    row = credential_row(["token = get_token()"], line_start=1, line_end=1, value_start=None, value_end=None)
    assert row["entities"] == []


def test_a_title_is_stripped_and_a_bare_surname_is_dropped():
    from external import without_title

    full = {"span_text": "Mrs Eileen Appleby", "start_offset": 10, "end_offset": 28}
    assert without_title(full) == {"start": 14, "end": 28, "label": "person"}
    assert without_title({"span_text": "Mrs Appleby", "start_offset": 0, "end_offset": 11}) is None
    assert without_title({"span_text": "Rolf Lundevall", "start_offset": 0, "end_offset": 14})["start"] == 0
