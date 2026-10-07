# /// script
# requires-python = ">=3.12"
# dependencies = ["pyarrow"]
# ///
"""Extract Traditional Chinese paragraphs from the zh Wikipedia shard, to rank vocabulary.

usage: uv run scripts/traditional_text.py
Writes data/wikimedia/wiki_traditional/texts.jsonl, one {"text"} per paragraph. The prune
ranked its 64,000 pieces on template text, which holds little real Traditional writing,
and tw-PII-bench then needs 19% more tokens than the full vocabulary. Wikipedia is no
evaluation set; Wikinews, which is, is a separate project.
"""

import json
from pathlib import Path

from pyarrow import parquet

ROOT = Path(__file__).resolve().parent.parent
SHARD = ROOT / "data/wikimedia/wikipedia/20231101.zh/train-00002-of-00006.parquet"
OUT = ROOT / "data/wikimedia/wiki_traditional/texts.jsonl"
# Common characters that differ between the scripts, paired by position.
TRADITIONAL = set("們個這來為說時會對國學與還後過開關義點發經無長東車門問間實現電話號碼業應當將從頭兩麼寫邊區麗變體價")
SIMPLIFIED = set("们个这来为说时会对国学与还后过开关义点发经无长东车门问间实现电话号码业应当将从头两么写边区丽变体价")
MIN_MARKERS = 3
MIN_CHARS = 30


def is_traditional(text: str) -> bool:
    traditional = sum(character in TRADITIONAL for character in text)
    return traditional >= MIN_MARKERS and not any(character in SIMPLIFIED for character in text)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    articles = parquet.read_table(SHARD, columns=["text"]).column("text").to_pylist()
    kept = 0
    with open(OUT, "w", encoding="utf-8") as handle:
        for article in articles:
            for paragraph in article.split("\n"):
                if len(paragraph) >= MIN_CHARS and is_traditional(paragraph):
                    handle.write(json.dumps({"text": paragraph}, ensure_ascii=False) + "\n")
                    kept += 1
    print(f"{kept} Traditional paragraphs from {len(articles)} articles -> {OUT}")


if __name__ == "__main__":
    main()
