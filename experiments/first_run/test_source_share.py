import random
from collections import Counter

from train import boosted_picker


def test_without_shares_the_draw_matches_a_plain_choice():
    rows = [{"source": "a", "n": i} for i in range(50)]
    plain = random.Random(3)
    expected = [plain.choice(rows) for _ in range(20)]
    draw = boosted_picker(rows, {}, random.Random(3))
    assert [draw() for _ in range(20)] == expected


def test_a_small_source_gets_its_share_of_draws():
    rows = [{"source": "big"} for _ in range(9900)] + [{"source": "small"} for _ in range(100)]
    draw = boosted_picker(rows, {"small": 0.2}, random.Random(5))
    counts = Counter(draw()["source"] for _ in range(20000))
    assert abs(counts["small"] / 20000 - 0.2) < 0.02


def test_a_share_for_a_source_absent_from_the_bucket_changes_nothing_but_the_rolls():
    rows = [{"source": "big"} for _ in range(10)]
    draw = boosted_picker(rows, {"missing": 0.5}, random.Random(1))
    assert {draw()["source"] for _ in range(100)} == {"big"}
