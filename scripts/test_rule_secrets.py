import random

from rule_secrets import CONTEXTS, PLACEHOLDERS, generate, provider_value, rule_ids, rules_detect


def spans_slice_to_values(row):
    return all(row["text"][s["start"] : s["end"]] == s["value"] for s in row["spans"])


def test_a_provider_value_is_the_secret_group_of_its_rule():
    generator = random.Random(1)
    produced = [provider_value(rule, generator) for rule in rule_ids(held=False)[:40]]
    produced = [p for p in produced if p]
    assert len(produced) >= 30
    for line, start, end in produced:
        assert 0 <= start < end <= len(line)


def test_every_context_renders_spans_that_slice_to_their_values():
    generator = random.Random(2)
    entries = [("api_key", "sk-test-9f8e7d6c5b4a", True), ("region", "ap-northeast-1", False)]
    for render in CONTEXTS.values():
        text, spans = render(entries, generator)
        row = {"text": text, "spans": spans}
        assert spans_slice_to_values(row), render.__name__
        assert [s["value"] for s in spans] == ["sk-test-9f8e7d6c5b4a"]


def test_generated_rows_slice_certify_and_never_label_a_placeholder():
    rows = list(generate(300, held=False, seed=3))
    assert len(rows) == 300
    for row in rows:
        assert spans_slice_to_values(row)
        assert not any(s["value"] in PLACEHOLDERS for s in row["spans"])
        for span in row["spans"]:
            if span.get("rule"):
                assert rules_detect(row["text"], span), span["rule"]


def test_held_out_providers_never_reach_the_train_rows():
    held = set(rule_ids(held=True))
    rows = list(generate(300, held=False, seed=4))
    assert not {s.get("rule") for row in rows for s in row["spans"]} & held
    held_rows = list(generate(50, held=True, seed=5))
    assert {s.get("rule") for row in held_rows for s in row["spans"] if s.get("rule")} <= held
