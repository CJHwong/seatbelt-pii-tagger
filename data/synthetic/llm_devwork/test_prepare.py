from pathlib import Path


def capped():
    # prepare.py imports the casual-text pipeline; load only the cap for a fast test.
    source = (Path(__file__).parent / "prepare.py").read_text()
    start = source.index("NEGATIVE_SHARE = ")
    end = source.index("def main")
    namespace = {"hashlib": __import__("hashlib")}
    exec(source[start:end], namespace)  # noqa: S102
    return namespace["capped"]


def row(row_id, positive):
    return {"id": row_id, "spans": [{"start": 0, "end": 1, "label": "person"}] if positive else []}


def test_negatives_are_capped_at_the_share_and_positives_all_stay():
    rows = [row(f"P{i}", True) for i in range(17)] + [row(f"N{i}", False) for i in range(20)]
    kept = capped()(rows)
    assert sum(1 for r in kept if r["spans"]) == 17
    assert sum(1 for r in kept if not r["spans"]) == 3  # 17 * 0.15 / 0.85


def test_an_ignore_only_row_counts_as_negative():
    rows = [row("P0", True), {"id": "I0", "spans": [{"start": 0, "end": 1, "label": "ignore"}]}]
    assert [r["id"] for r in capped()(rows)] == ["P0"]
