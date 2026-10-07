from seatbelt import held_out, occurrences, row_label_hit


def test_every_occurrence_of_a_value_is_marked_even_inside_a_word():
    text = 'KEY=abc123 and again:abc123"'
    assert occurrences(text, "abc123") == [
        {"start": 4, "end": 10, "label": "secret"},
        {"start": 21, "end": 27, "label": "secret"},
    ]


def test_the_held_out_split_is_stable_and_about_one_in_five():
    rules = [f"rule-{number}" for number in range(1000)]
    first = {rule for rule in rules if held_out(rule)}
    assert first == {rule for rule in rules if held_out(rule)}
    assert 150 < len(first) < 250


def test_a_row_label_hit_needs_the_expected_label_after_mapping():
    spans = [{"start": 0, "end": 3, "label": "private_email"}]
    assert row_label_hit(["private_email"], spans)
    assert not row_label_hit(["private_phone"], spans)
    assert row_label_hit(["private_email"], [{"start": 0, "end": 3, "label": "email"}])
