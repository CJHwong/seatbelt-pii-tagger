"""Every data source and sample the experiment uses, in one place.

Training and the pruning frequencies come from Wismut train only. Evaluation reads
Wismut test, tw-PII-bench and pii-bench-zh, which never feed training or pruning.
"""

import ast
import json
import random
import re
from functools import cache
from pathlib import Path

from pyarrow import parquet

DATA = Path(__file__).resolve().parents[2] / "data"
MINILM_REPO = "microsoft/Multilingual-MiniLM-L12-H384"
MINILM_REVISION = "6e8c1ec6b4ec4e3fc6eb7d2cd834fcd582b61daf"


@cache
def minilm() -> str:
    """The pinned base model's folder in the Hugging Face cache, downloaded on first use."""
    from huggingface_hub import snapshot_download

    return snapshot_download(MINILM_REPO, revision=MINILM_REVISION)


WISMUT = DATA / "Wismut/nym-pii-multilingual-data"
TW_BENCH = DATA / "lianghsun/tw-PII-bench/data"
ZH_BENCH = DATA / "wan9yu/pii-bench-zh/data/pii_bench_zh.jsonl"

LATIN_TRAIN_ROWS = 40_000
EVAL_ROWS_PER_SCRIPT = 1_500
ZH_BENCH_ROWS = 2_000
SEED = 13

HAN = re.compile(r"[一-鿿]")
KANA_OR_HANGUL = re.compile(r"[぀-ヿ가-힯]")
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LATIN = re.compile(r"[A-Za-z]")


def script_of(text: str) -> str | None:
    """'zh', 'latin', or None for scripts outside the English and Chinese target."""
    if KANA_OR_HANGUL.search(text) or CYRILLIC.search(text):
        return None
    if HAN.search(text):
        return "zh"
    return "latin" if LATIN.search(text) else None


def _read_wismut(split: str) -> dict[str, list[dict]]:
    by_script: dict[str, list[dict]] = {"latin": [], "zh": []}
    with open(WISMUT / f"{split}.jsonl") as handle:
        for line in handle:
            row = json.loads(line)
            script = script_of(row["text"])
            if script:
                by_script[script].append({"text": row["text"], "entities": row["entities"], "script": script})
    return by_script


@cache
def _wismut_train() -> dict[str, list[dict]]:
    return _read_wismut("train")


def pruning_texts() -> list[str]:
    """The whole train split. The training sample alone touches too few pieces to rank 64,000."""
    rows = _wismut_train()
    return [row["text"] for row in rows["latin"] + rows["zh"]]


@cache
def training_rows() -> list[dict]:
    rows = _wismut_train()
    latin = random.Random(SEED).sample(rows["latin"], LATIN_TRAIN_ROWS)
    return latin + rows["zh"]


@cache
def wismut_eval_rows() -> list[dict]:
    rows = _read_wismut("test")
    picker = random.Random(SEED)
    return picker.sample(rows["latin"], EVAL_ROWS_PER_SCRIPT) + picker.sample(rows["zh"], EVAL_ROWS_PER_SCRIPT)


@cache
def tw_bench_rows() -> list[dict]:
    rows = []
    for part in ("short", "mid", "long"):
        for row in parquet.read_table(TW_BENCH / f"{part}.parquet").to_pylist():
            spans = row["spans"] if isinstance(row["spans"], list) else ast.literal_eval(row["spans"])
            entities = [{"start": s["start"], "end": s["end"], "label": s["label"]} for s in spans]
            rows.append({"text": row["text"], "entities": entities, "negative": str(row["is_negative"]) == "True"})
    return rows


@cache
def zh_bench_rows() -> list[dict]:
    with open(ZH_BENCH) as handle:
        rows = [json.loads(line) for line in handle]
    picked = random.Random(SEED).sample(rows, ZH_BENCH_ROWS)
    return [
        {
            "text": row["text"],
            "entities": [{"start": e["start"], "end": e["end"], "label": e["type"]} for e in row["entities"]],
        }
        for row in picked
    ]


def entity_types() -> list[str]:
    return sorted({entity["label"] for row in training_rows() for entity in row["entities"]})


def eval_texts() -> dict[str, list[str]]:
    return {
        "wismut_test_latin": [row["text"] for row in wismut_eval_rows() if row["script"] == "latin"],
        "wismut_test_zh": [row["text"] for row in wismut_eval_rows() if row["script"] == "zh"],
        "tw_bench": [row["text"] for row in tw_bench_rows()],
        "zh_bench": [row["text"] for row in zh_bench_rows()],
    }
