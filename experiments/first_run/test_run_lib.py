from run_lib import (
    bench_gold,
    drop_name_repeats,
    drop_overlapping,
    is_birth_date,
    layer_scale,
    length_batches,
    mask_ignored,
    per_type_counts,
    script_bucket,
)


def test_birth_cue_on_either_side_marks_a_birth_date():
    assert is_birth_date("出生日期：1990/01/02", 5, 15)
    assert is_birth_date("我是1990年1月2日出生的", 2, 11)
    assert not is_birth_date("出院日期：2024/03/05", 5, 15)


def test_bench_gold_maps_types_and_sets_other_dates_aside():
    text = "王小明，生日1990/01/02，出院日期2024/03/05"
    spans = [
        {"start": 0, "end": 3, "label": "private_person"},
        {"start": 6, "end": 16, "label": "private_date"},
        {"start": 21, "end": 31, "label": "private_date"},
    ]
    entities, dont_care = bench_gold(text, spans)
    assert entities == [{"start": 0, "end": 3, "label": "person"}, {"start": 6, "end": 16, "label": "date"}]
    assert dont_care == [{"start": 21, "end": 31}]


def test_line_and_ptt_ids_score_as_username():
    entities, _ = bench_gold(
        "ab", [{"start": 0, "end": 1, "label": "tw_line_id"}, {"start": 1, "end": 2, "label": "tw_ptt_id"}]
    )
    assert [e["label"] for e in entities] == ["username", "username"]


def test_predictions_on_a_dont_care_span_are_dropped():
    predicted = [{"start": 0, "end": 3, "label": "date"}, {"start": 5, "end": 8, "label": "person"}]
    assert drop_overlapping(predicted, [{"start": 2, "end": 4}]) == [predicted[1]]


def test_mask_ignored_skips_tokens_inside_ignore_spans_only():
    tags = ["O", "B-person", "O", None]
    spans = [(0, 2), (2, 4), (4, 6), None]
    assert mask_ignored(tags, spans, [{"start": 3, "end": 5}]) == ["O", None, None, None]


def test_per_type_counts_split_hits_by_label():
    gold = [{"start": 0, "end": 2, "label": "person"}, {"start": 3, "end": 5, "label": "phone"}]
    predicted = [{"start": 0, "end": 2, "label": "person"}, {"start": 3, "end": 5, "label": "email"}]
    counts = per_type_counts(gold, predicted)
    assert counts["person"] == {"tp": 1, "fp": 0, "fn": 0}
    assert counts["phone"] == {"tp": 0, "fp": 0, "fn": 1}
    assert counts["email"] == {"tp": 0, "fp": 1, "fn": 0}


def test_script_bucket_folds_unmarked_chinese_into_simplified():
    assert [script_bucket(s) for s in ("latin", "zh", "zh-Hans", "zh-Hant")] == [
        "latin",
        "zh-Hans",
        "zh-Hans",
        "zh-Hant",
    ]


def test_later_mentions_of_a_gold_name_are_set_aside():
    text = "王小明說，小明明天到。王小明先生好。陳大文來了。"
    gold = [{"start": 0, "end": 3, "label": "person"}]
    predicted = [
        {"start": 0, "end": 3, "label": "person"},
        {"start": 5, "end": 7, "label": "person"},
        {"start": 11, "end": 16, "label": "person"},
        {"start": 18, "end": 21, "label": "person"},
    ]
    kept, set_aside = drop_name_repeats(predicted, text, gold)
    assert kept == [predicted[0], predicted[3]]
    assert set_aside == 2


def test_length_batches_groups_similar_lengths_and_keeps_every_row():
    import random

    lengths = [5, 50, 6, 51, 7, 52, 8, 53]
    batches = length_batches(lengths, 4, random.Random(0))
    assert sorted(index for batch in batches for index in batch) == list(range(8))
    assert sorted(sorted(lengths[i] for i in batch) for batch in batches) == [[5, 6, 7, 8], [50, 51, 52, 53]]


def test_layer_scale_lowers_the_rate_one_factor_per_layer_down():
    # ULMFiT's discriminative rates: the top layer and the head train at the full rate
    assert layer_scale("classifier.weight", 6, 0.9) == 1.0
    assert layer_scale("bert.encoder.layer.5.output.dense.weight", 6, 0.9) == 1.0
    assert layer_scale("bert.encoder.layer.4.attention.self.query.bias", 6, 0.9) == 0.9
    assert abs(layer_scale("bert.encoder.layer.0.output.dense.weight", 6, 0.9) - 0.9**5) < 1e-12
    assert abs(layer_scale("bert.embeddings.word_embeddings.weight", 6, 0.9) - 0.9**6) < 1e-12


def test_layer_scale_of_one_keeps_every_rate():
    for name in ("classifier.bias", "bert.encoder.layer.0.output.dense.weight", "bert.embeddings.LayerNorm.weight"):
        assert layer_scale(name, 6, 1.0) == 1.0
