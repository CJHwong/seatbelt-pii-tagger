import json
import math
from collections import Counter

import pytest

from prune_lib import (
    SPACE_MARK,
    allowed_transition,
    bioes_labels,
    boundary_misaligned,
    decode_spans,
    effective_spans,
    f1_report,
    join_touching,
    merge_adjacent,
    news_filter,
    office_filter,
    overlap_counts,
    place_filter,
    post_rules,
    pruned_tokenizer_json,
    public_filter,
    public_url,
    select_pieces,
    span_counts,
    token_tags,
    tool_rules,
    username_keys,
    viterbi,
    with_punctuation_split,
)


def test_bioes_labels_has_one_plus_four_per_type():
    labels = bioes_labels(["EMAIL", "PHONE"])
    assert labels == ["O", "B-EMAIL", "I-EMAIL", "E-EMAIL", "S-EMAIL", "B-PHONE", "I-PHONE", "E-PHONE", "S-PHONE"]


def test_bare_space_mark_covers_nothing():
    # Real tokenizer output: the bare mark before 張 carries the offset of 張.
    tokens = ["<s>", SPACE_MARK, "張", "三", "</s>"]
    offsets = [(0, 0), (0, 1), (0, 1), (1, 2), (0, 0)]
    assert effective_spans(tokens, offsets) == [None, None, (0, 1), (1, 2), None]


def test_token_tags_multi_and_single_token_entities():
    spans = [None, (0, 1), (1, 2), (2, 3), (4, 9), None]
    entities = [{"start": 0, "end": 2, "label": "NAME"}, {"start": 4, "end": 9, "label": "EMAIL"}]
    assert token_tags(spans, entities) == [None, "B-NAME", "E-NAME", "O", "S-EMAIL", None]


def test_token_tags_three_tokens_get_inside():
    spans = [(0, 2), (2, 4), (4, 6)]
    assert token_tags(spans, [{"start": 0, "end": 6, "label": "ID"}]) == ["B-ID", "I-ID", "E-ID"]


def test_boundary_misaligned_when_entity_splits_a_token():
    # 電話 is one token; an entity covering only 電 cannot be represented exactly.
    spans = [(0, 2), (2, 4)]
    assert boundary_misaligned(spans, {"start": 0, "end": 1, "label": "X"})
    assert not boundary_misaligned(spans, {"start": 0, "end": 4, "label": "X"})


def test_transitions_follow_bioes():
    assert allowed_transition("B-A", "I-A")
    assert allowed_transition("B-A", "E-A")
    assert not allowed_transition("B-A", "O")
    assert not allowed_transition("B-A", "E-B")
    assert not allowed_transition("O", "I-A")
    assert allowed_transition("E-A", "B-B")
    assert allowed_transition("S-A", "O")


def test_viterbi_repairs_an_illegal_argmax():
    labels = bioes_labels(["A"])  # O, B-A, I-A, E-A, S-A
    # Argmax per token is B-A then O, which is illegal. The legal best is B-A, E-A.
    rows = [
        [math.log(0.1), math.log(0.8), math.log(0.02), math.log(0.02), math.log(0.06)],
        [math.log(0.5), math.log(0.02), math.log(0.02), math.log(0.44), math.log(0.02)],
    ]
    # B-A, E-A scores log 0.8 + log 0.44 = -1.04, ahead of O, O at -3.00.
    assert [labels[index] for index in viterbi(rows, labels)] == ["B-A", "E-A"]


def test_viterbi_never_ends_inside_an_entity():
    labels = bioes_labels(["A"])
    rows = [[-5, -0.1, -5, -5, -5], [-5, -5, -0.1, -5, -5]]  # wants B-A then I-A
    path = [labels[index] for index in viterbi(rows, labels)]
    assert path[-1] in ("O", "E-A", "S-A")


def test_decode_spans_from_tags():
    tags = ["B-NAME", "E-NAME", "O", "S-EMAIL"]
    spans = [(0, 1), (1, 3), (3, 4), (5, 9)]
    assert decode_spans(tags, spans) == [
        {"start": 0, "end": 3, "label": "NAME"},
        {"start": 5, "end": 9, "label": "EMAIL"},
    ]


def test_decode_spans_scores_each_span_by_its_least_sure_token():
    tags = ["B-NAME", "E-NAME", "O", "S-EMAIL"]
    spans = [(0, 1), (1, 3), (3, 4), (5, 9)]
    assert decode_spans(tags, spans, confidence=[0.9, 0.5, 0.1, 0.8]) == [
        {"start": 0, "end": 3, "label": "NAME", "score": 0.5},
        {"start": 5, "end": 9, "label": "EMAIL", "score": 0.8},
    ]


def test_join_touching_joins_secret_pieces_but_not_touching_names():
    spans = [
        {"start": 400, "end": 632, "label": "secret", "score": 0.6},
        {"start": 19, "end": 400, "label": "secret", "score": 0.9},
        {"start": 632, "end": 634, "label": "person", "score": 1.0},
        {"start": 634, "end": 636, "label": "person", "score": 1.0},
    ]
    assert join_touching(spans) == [
        {"start": 19, "end": 632, "label": "secret", "score": 0.6},
        {"start": 632, "end": 634, "label": "person", "score": 1.0},
        {"start": 634, "end": 636, "label": "person", "score": 1.0},
    ]


def _span(text: str, part: str, label: str) -> dict:
    start = text.index(part)
    return {"start": start, "end": start + len(part), "label": label, "score": 0.9}


def test_post_rules_drop_a_chinese_city_or_district_alone_but_keep_street_addresses():
    text = "他住在台北市大安區，公司在台北市信義區松仁路100號5樓，辦公室在C座，老家在 Jalan Muhibbah."
    spans = [
        _span(text, "台北市大安區", "address"),
        _span(text, "台北市信義區松仁路100號5樓", "address"),
        _span(text, "C座", "address"),
        _span(text, "Jalan Muhibbah", "address"),
    ]
    assert [text[s["start"] : s["end"]] for s in post_rules(spans, text)] == [
        "台北市信義區松仁路100號5樓",
        "C座",
        "Jalan Muhibbah",
    ]


def test_post_rules_drop_public_host_urls_but_keep_profiles_and_userinfo():
    text = "see https://docs.python.org/3/ and https://example.com/a, me at https://github.com/users/ann or https://bob:pw@pypi.org/x"
    spans = [
        _span(text, "https://docs.python.org/3/", "url"),
        _span(text, "https://example.com/a", "url"),
        _span(text, "https://github.com/users/ann", "url"),
        _span(text, "https://bob:pw@pypi.org/x", "url"),
    ]
    assert [text[s["start"] : s["end"]] for s in post_rules(spans, text)] == [
        "https://github.com/users/ann",
        "https://bob:pw@pypi.org/x",
    ]


def test_post_rules_drop_shared_mailboxes_and_default_accounts():
    text = "mail service@shop.tw, ann@example.org or ann.lee@gmail.com; login root or annlee88"
    spans = [
        _span(text, "service@shop.tw", "email"),
        _span(text, "ann@example.org", "email"),
        _span(text, "ann.lee@gmail.com", "email"),
        _span(text, "root", "username"),
        _span(text, "annlee88", "username"),
    ]
    assert [text[s["start"] : s["end"]] for s in post_rules(spans, text)] == [
        "ann@example.org",
        "ann.lee@gmail.com",
        "annlee88",
    ]


def test_post_rules_drop_a_username_inside_a_dropped_url():
    text = "clone https://pypi.org/project/requests now"
    spans = [_span(text, "https://pypi.org/project/requests", "url"), _span(text, "requests", "username")]
    assert post_rules(spans, text) == []


PUBLIC = {"歐巴馬", "barack obama", "蔡英文"}


def test_public_filter_drops_a_listed_full_name_and_its_later_surname():
    text = "Barack Obama spoke today. Later Obama met 蔡英文 and Ann Lee."
    spans = [
        _span(text, "Barack Obama", "person"),
        _span(text, "Obama met", "person") | {"end": text.index("Obama met") + 5},
        _span(text, "蔡英文", "person"),
        _span(text, "Ann Lee", "person"),
    ]
    assert [text[s["start"] : s["end"]] for s in public_filter(spans, text, PUBLIC)] == ["Ann Lee"]


def test_public_filter_keeps_a_listed_name_next_to_other_pii():
    text = "蔡英文 的手機 0912345678"
    spans = [_span(text, "蔡英文", "person"), _span(text, "0912345678", "phone")]
    assert public_filter(spans, text, PUBLIC) == spans


def test_public_filter_ignores_a_single_word_or_character_match():
    text = "Obama and 蔡 came"
    spans = [_span(text, "Obama", "person"), _span(text, "蔡", "person")]
    assert public_filter(spans, text, PUBLIC | {"obama", "蔡"}) == spans


def test_place_filter_drops_person_spans_that_name_a_listed_place():
    text = "德國與菲律賓的代表王小明，以及 Fiji 和 New Zealand"
    spans = [
        _span(text, "德國", "person"),
        _span(text, "菲律賓", "person"),
        _span(text, "王小明", "person"),
        _span(text, "Fiji", "person"),
        _span(text, "New Zealand", "person"),
    ]
    places = {"德國", "菲律賓", "fiji", "new zealand"}
    assert [text[s["start"] : s["end"]] for s in place_filter(spans, text, places)] == [
        "王小明",
        "Fiji",
        "New Zealand",
    ]


def test_merge_adjacent_joins_name_parts_across_one_space():
    text = "Ann Lee called"
    spans = [{"start": 0, "end": 3, "label": "GIVEN"}, {"start": 4, "end": 7, "label": "SURNAME"}]
    assert merge_adjacent(spans, text) == [{"start": 0, "end": 7, "label": "PII"}]


def test_merge_adjacent_keeps_spans_split_by_text():
    text = "Ann and Lee"
    spans = [{"start": 0, "end": 3, "label": "A"}, {"start": 8, "end": 11, "label": "A"}]
    assert len(merge_adjacent(spans, text)) == 2


def test_span_counts_typed_and_untyped():
    gold = [{"start": 0, "end": 3, "label": "A"}, {"start": 5, "end": 8, "label": "B"}]
    predicted = [{"start": 0, "end": 3, "label": "B"}, {"start": 9, "end": 10, "label": "A"}]
    assert span_counts(gold, predicted, typed=True) == Counter(tp=0, fp=2, fn=2)
    assert span_counts(gold, predicted, typed=False) == Counter(tp=1, fp=1, fn=1)


def test_overlap_counts_and_report():
    gold = [{"start": 0, "end": 5, "label": "A"}, {"start": 10, "end": 12, "label": "A"}]
    predicted = [{"start": 2, "end": 7, "label": "A"}]
    counts = overlap_counts(gold, predicted)
    assert counts == Counter(tp_gold=1, fn=1, tp_pred=1, fp=0)
    report = f1_report(counts)
    assert report["precision"] == 1.0 and report["recall"] == 0.5


def test_f1_report_handles_empty():
    assert f1_report(Counter(tp=0, fp=0, fn=0))["f1"] == 0.0


VOCAB = [
    ["<s>", 0.0],
    ["<pad>", 0.0],
    ["</s>", 0.0],
    ["<unk>", 0.0],
    ["a", -1.0],
    ["b", -1.0],
    ["ab", -2.0],
    ["ba", -2.0],
    ["abab", -3.0],
    ["<mask>", 0.0],
]


def test_select_pieces_forces_specials_and_single_characters():
    kept = select_pieces(VOCAB, Counter({8: 50, 7: 10, 6: 1}), target_size=8)
    # Forced: 4 specials, <mask>, a, b. One slot left, and "abab" is the most frequent.
    assert kept == [0, 1, 2, 3, 4, 5, 8, 9]


def test_select_pieces_rejects_a_target_below_the_forced_set():
    with pytest.raises(ValueError):
        select_pieces(VOCAB, Counter(), target_size=5)


def test_select_pieces_rejects_a_corpus_that_cannot_fill_the_target():
    with pytest.raises(ValueError):
        select_pieces(VOCAB, Counter({8: 1}), target_size=9)


def _tokenizer_json() -> str:
    return json.dumps(
        {
            "added_tokens": [
                {"id": 0, "content": "<s>"},
                {"id": 3, "content": "<unk>"},
                {"id": 9, "content": "<mask>"},
            ],
            "model": {"type": "Unigram", "unk_id": 3, "vocab": VOCAB},
            "post_processor": {
                "type": "TemplateProcessing",
                "special_tokens": {"<s>": {"id": "<s>", "ids": [0]}, "</s>": {"id": "</s>", "ids": [2]}},
            },
        }
    )


def test_pruned_tokenizer_json_renumbers_everything():
    kept = [0, 1, 2, 3, 4, 5, 8, 9]
    spec = json.loads(pruned_tokenizer_json(_tokenizer_json(), kept))
    assert [piece for piece, _ in spec["model"]["vocab"]] == [
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "a",
        "b",
        "abab",
        "<mask>",
    ]
    assert spec["model"]["unk_id"] == 3
    assert [added["id"] for added in spec["added_tokens"]] == [0, 3, 7]
    assert spec["post_processor"]["special_tokens"]["</s>"]["ids"] == [2]


def test_viterbi_numpy_matches_the_reference():
    import numpy

    from prune_lib import transition_matrix, viterbi_numpy

    labels = bioes_labels(["A", "B", "C"])
    transitions = transition_matrix(labels)
    generator = numpy.random.default_rng(7)
    for length in (1, 2, 5, 17):
        for _ in range(20):
            logits = generator.normal(size=(length, len(labels)))
            log_probs = logits - numpy.log(numpy.exp(logits).sum(axis=1, keepdims=True))
            assert viterbi_numpy(log_probs, labels, transitions) == viterbi(log_probs.tolist(), labels)


def test_chunk_text_packs_sentences_and_round_trips():
    from prune_lib import chunk_text

    text = "甲乙。丙丁戊。己\n庚辛壬癸子丑寅卯"
    chunks = chunk_text(text, max_chars=7)
    assert "".join(chunk for _, chunk in chunks) == text
    assert all(len(chunk) <= 7 for _, chunk in chunks)
    assert all(text[offset : offset + len(chunk)] == chunk for offset, chunk in chunks)
    assert chunks[0] == (0, "甲乙。丙丁戊。")


def test_chunk_text_cuts_an_overlong_sentence():
    from prune_lib import chunk_text

    assert chunk_text("abcdefghij", max_chars=4) == [(0, "abcd"), (4, "efgh"), (8, "ij")]


def test_shifted_moves_spans():
    from prune_lib import shifted

    assert shifted([{"start": 1, "end": 3, "label": "A"}], 10) == [{"start": 11, "end": 13, "label": "A"}]


def test_split_punctuation_isolates_punctuation_from_digits():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Metaspace, Sequence, WhitespaceSplit

    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Sequence([WhitespaceSplit(), Metaspace()])
    split = Tokenizer.from_str(with_punctuation_split(tokenizer.to_str()))
    pieces = [piece for piece, _ in split.pre_tokenizer.pre_tokenize_str("電話:0912-345 (手機)")]
    assert pieces == ["▁電話", ":", "0912", "-", "345", "▁", "(", "手機", ")"]


def test_viterbi_numpy_takes_a_legal_greedy_path_and_matches_the_reference():
    import numpy

    from prune_lib import transition_matrix, viterbi_numpy

    labels = bioes_labels(["A", "B"])
    transitions = transition_matrix(labels)
    generator = numpy.random.default_rng(11)
    legal = ["O", "B-A", "I-A", "E-A", "S-B", "O", "B-B", "E-B"]
    for _ in range(20):
        logits = generator.normal(size=(len(legal), len(labels)))
        for position, tag in enumerate(legal):
            logits[position, labels.index(tag)] += 8.0
        log_probs = logits - numpy.log(numpy.exp(logits).sum(axis=1, keepdims=True))
        path = viterbi_numpy(log_probs, labels, transitions)
        assert [labels[index] for index in path] == legal
        assert path == viterbi(log_probs.tolist(), labels)


def test_public_url_keeps_an_unparsable_url_flagged():
    assert public_url("http://[docs.python.org/x") is False


def test_calibrated_spans_map_each_label_and_fall_back_to_the_global_map():
    from prune_lib import calibrated_spans

    maps = {"person": [2.0, 0.0], "*": [1.0, -1.0]}
    spans = [
        {"label": "person", "start": 0, "end": 2, "score": 0.5},
        {"label": "phone", "start": 3, "end": 9, "score": 0.5},
    ]
    person, phone = calibrated_spans(spans, maps)
    assert person == {"label": "person", "start": 0, "end": 2, "score": 0.5}
    assert abs(phone["score"] - 1 / (1 + math.e)) < 1e-12
    assert calibrated_spans(spans, None) == spans


def _persons(text: str, *names: str) -> list[dict]:
    spans, start = [], 0
    for name in names:
        start = text.index(name, start)
        spans.append({"start": start, "end": start + len(name), "label": "person", "score": 0.9})
        start += len(name)
    return spans


def test_office_filter_drops_a_titled_name_and_its_later_mentions():
    text = "立法會議員郭榮鏗今日表示反對。郭榮鏗又說，陳怡君的看法不同。"
    spans = _persons(text, "郭榮鏗", "郭榮鏗", "陳怡君")
    kept = office_filter(spans, text)
    assert [text[s["start"] : s["end"]] for s in kept] == ["陳怡君"]


def test_office_filter_reads_titles_after_the_name_and_english_titles():
    text = "彭斯副總統 met Senator John Smith and jake."
    spans = _persons(text, "彭斯", "John Smith", "jake")
    assert [text[s["start"] : s["end"]] for s in office_filter(spans, text)] == ["jake"]


def test_office_filter_keeps_a_titled_name_next_to_private_pii():
    text = "議員王小明 電話0912345678"
    spans = _persons(text, "王小明") + [{"start": 9, "end": 19, "label": "phone", "score": 0.9}]
    assert office_filter(spans, text) == spans


def test_news_filter_drops_names_in_a_news_register_without_private_pii():
    text = "（中央社記者王小明台北2日電）行政院發言人今天表示，陳大文的提案已送交立法院。記者李小華報導。"
    spans = _persons(text, "王小明", "陳大文", "李小華")
    assert news_filter(spans, text) == []


def test_news_filter_keeps_names_next_to_private_pii_and_leaves_other_registers_alone():
    filler = "警方表示案情仍在調查中，相關單位將持續追蹤。" * 12
    news = "記者王小明報導：" + filler + "受害人陳大文（電話0912345678）昨晚報案。"
    spans = _persons(news, "王小明", "陳大文") + [
        {"start": news.index("0912"), "end": news.index("0912") + 10, "label": "phone", "score": 0.9}
    ]
    kept = news_filter(spans, news)
    assert [news[s["start"] : s["end"]] for s in kept if s["label"] == "person"] == ["陳大文"]
    chat = "欸 陳大文 你明天要不要來 我跟王小明約了"
    chat_spans = _persons(chat, "陳大文", "王小明")
    assert news_filter(chat_spans, chat) == chat_spans


def _spans_of(text: str, label: str, *values: str) -> list[dict]:
    spans, start = [], 0
    for value in values:
        start = text.index(value, start)
        spans.append({"start": start, "end": start + len(value), "label": label, "score": 0.9})
        start += len(value)
    return spans


def test_post_rules_drop_code_identifiers_flagged_as_usernames_but_keep_handles():
    text = "x = nx.Graph()\nhidden_size = 4\nassert_frame_equal(a, b)\nclass Foo:\nmail from jeanette.blair and my venmo is jake_99"
    spans = _spans_of(
        text, "username", "nx.Graph", "hidden_size", "assert_frame_equal", "Foo", "jeanette.blair", "jake_99"
    )
    kept = [text[s["start"] : s["end"]] for s in post_rules(spans, text)]
    assert kept == ["jeanette.blair", "jake_99"]


def test_post_rules_drop_repository_and_documentation_urls_but_keep_profiles():
    text = (
        "see https://github.com/huggingface/transformers/issues/1 and http://www.apache.org/licenses/LICENSE-2.0 "
        "or docs/source/en/fusion_mapping.md; my page is https://github.com/lumina37 and https://www.pixnet.net/blog/me"
    )
    spans = _spans_of(
        text,
        "url",
        "https://github.com/huggingface/transformers/issues/1",
        "http://www.apache.org/licenses/LICENSE-2.0",
        "docs/source/en/fusion_mapping.md",
        "https://github.com/lumina37",
        "https://www.pixnet.net/blog/me",
    )
    kept = [text[s["start"] : s["end"]] for s in post_rules(spans, text)]
    assert kept == ["https://github.com/lumina37", "https://www.pixnet.net/blog/me"]


def test_tool_rules_drop_line_numbers_and_keep_numbers_in_prose():
    listing = "   134\t        amount=price,\n   135\t        currency=usd"
    reference = "see app/models/accounting.py:4821 for the charge line"
    prose = "call me at 4821 tonight"
    assert tool_rules(_spans_of(listing, "address", "134"), listing) == []
    assert tool_rules(_spans_of(reference, "secret", "4821"), reference) == []
    assert tool_rules(_spans_of(prose, "secret", "4821"), prose) == _spans_of(prose, "secret", "4821")


def test_tool_rules_drop_json_keys_scoped_names_and_headings():
    keyed = '{"verdict": "near", "reason": "It concerns the fix"}'
    scoped = "if mode == Mode::Json { format = Format::Text }"
    heading = "Context.\nWHAT THE USER ASKED FOR:\ndid it ship"
    assert tool_rules(_spans_of(keyed, "person", "reason"), keyed) == []
    assert tool_rules(_spans_of(scoped, "person", "Json", "Text"), scoped) == []
    assert tool_rules(_spans_of(heading, "person", "WHAT THE USER ASKED FOR"), heading) == []


def test_tool_rules_keep_a_name_that_only_looks_like_code():
    signed = 'Reviewed by "Gary Lin": looks good'
    shouting = "Thanks to JOHN SMITH for the fix"
    assert tool_rules(_spans_of(signed, "person", "Gary Lin"), signed) == _spans_of(signed, "person", "Gary Lin")
    assert tool_rules(_spans_of(shouting, "person", "JOHN SMITH"), shouting) == _spans_of(
        shouting, "person", "JOHN SMITH"
    )


def test_tool_rules_drop_hashes_in_logs_and_paths_but_keep_bare_keys():
    log = "4f2a9c1 fix(ops): pin the probe\n8be01d3 test: add a case"
    path = "/tmp/projects/toolkit/0b7e3a52-91cd-4f0e-9a1b-5c2d8e6f7a10/subagents/agent.jsonl"
    bare = "TOKEN=0b7e3a52-91cd-4f0e-9a1b-5c2d8e6f7a10"
    digits = "1234567890 national id on file"
    assert tool_rules(_spans_of(log, "secret", "4f2a9c1", "8be01d3"), log) == []
    assert tool_rules(_spans_of(path, "secret", "0b7e3a52-91cd-4f0e-9a1b-5c2d8e6f7a10"), path) == []
    assert tool_rules(_spans_of(bare, "secret", "0b7e3a52-91cd-4f0e-9a1b-5c2d8e6f7a10"), bare) == _spans_of(
        bare, "secret", "0b7e3a52-91cd-4f0e-9a1b-5c2d8e6f7a10"
    )
    assert tool_rules(_spans_of(digits, "national_id", "1234567890"), digits) == _spans_of(
        digits, "national_id", "1234567890"
    )


def _values(spans, text):
    return [(text[s["start"] : s["end"]], s["label"]) for s in spans]


def test_username_keys_label_a_handle_after_a_username_key():
    roster = "- { name: Jo Lee, slack: U011ABCDE9, github: jlee-42 }"
    log = "PR#12 | author=mchen | reviewer=t.wu | state=open"
    assert _values(username_keys([], roster), roster) == [("U011ABCDE9", "username"), ("jlee-42", "username")]
    assert _values(username_keys([], log), log) == [("mchen", "username"), ("t.wu", "username")]


def test_username_keys_read_table_mentions_but_not_paths_or_account_keys():
    paths = "gh api /users/octocat/repos; ls /Users/jdoe/Documents and /home/deploy/app"
    accounts = "login: dbuser\nusername: svc_backup\nhandle: ops"
    table = "| Jo Lee | @jlee42 | backend |"
    assert username_keys([], paths) == []
    assert username_keys([], accounts) == []
    assert _values(username_keys([], table), table) == [("jlee42", "username")]


def test_username_keys_skip_placeholders_bots_and_covered_values():
    skipped = "author=<login> slack: TBD github: null author=dependabot[bot] reviewer=renovate-bot author: 12345"
    assert username_keys([], skipped) == []
    covered = "author=mchen"
    existing = [{"start": 7, "end": 12, "label": "person", "score": 0.9}]
    assert username_keys(existing, covered) == existing


def test_username_keys_skip_namespaced_commands_calls_code_and_review_bots():
    command = "run /gf-github:pr-create then `tool-gitlab:sync`"
    query = "assignee = currentUser() AND status = Done; assignee=currentUser()"
    code = "author = 'PlaintextProbe'\nreviewer=reviewer"
    bots = "reviewer=human reviewer=coderabbit author: copilot-pull-request-reviewer assignee=unassigned | @reviewer |"
    for text in (command, query, code, bots):
        assert username_keys([], text) == [], text
