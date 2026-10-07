# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers", "sentencepiece", "protobuf", "numpy", "pyarrow", "pyyaml"]
# ///
"""Choose and apply the public-name filters, `prune_lib.public_filter` and `place_filter`.

usage: uv run --script public_filter.py dev KIND FLOOR...           gold person names dropped
       uv run --script public_filter.py wiki KIND SYSTEM FLOOR...   person flags dropped on Wikipedia
       uv run --script public_filter.py apply KIND SYSTEM FLOOR     -> runs/[GROUP/]SYSTEM_SUFFIX.jsonl

KIND is persons (suffix _pub), han_persons (suffix _pubh), places (suffix _plc) or offices
(suffix _off). han_persons takes FLOOR for Han names only and keeps LATIN_FLOOR for the
rest: on the dev split the low floors cost Latin names (John Doe, David Gates) and almost
no Han ones. offices uses
no list, so its FLOOR is ignored: it drops names next to a public-office title. FLOOR is the sitelink count an item
needs, strictly more than, to enter the list. dev and wiki use no evaluation set: dev is
the training sources' dev split, and wiki is 16,008 mined Wikipedia paragraphs, which no
evaluation set shares and training does not use. Choose the floor on
those two, then apply it once.
"""

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path[:0] = [str(ROOT / "experiments/first_run"), str(ROOT / "experiments/vocab_prune"), str(HERE)]

from prune_lib import name_key, news_filter, office_filter, place_filter, public_filter

GAZETTEER = ROOT / "data/gazetteer"
WIKI = ROOT / "data/wikimedia/wiki_negatives/train.jsonl"
GROUPS = ["comparison", "generalization", "blind", "rules", "benchmarks", "published"]
SUFFIXES = {"persons": "_pub", "han_persons": "_pubh", "places": "_plc", "offices": "_off", "news": "_news"}
LATIN_FLOOR = 20
HAN_CHARACTER = re.compile(r"[\u4e00-\u9fff]")


@cache
def holders(kind: str) -> tuple[dict[str, set[str]], dict[str, int]]:
    """Every name's Wikidata items, and every item's sitelink count."""
    names: dict[str, set[str]] = defaultdict(set)
    sitelinks: dict[str, int] = {}
    with open(GAZETTEER / f"public_{kind}.tsv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
            names[name_key(row["name"])].add(row["item"])
            sitelinks[row["item"]] = int(row["sitelinks"])
    return names, sitelinks


def listed(kind: str, floor: int) -> set[str]:
    """Persons: a name only one listed person holds, so a common name stays flagged.
    Places: a name of any place above the floor that no listed person holds."""
    if kind in {"offices", "news"}:
        return set()
    if kind == "han_persons":
        return listed("persons", LATIN_FLOOR) | {key for key in listed("persons", floor) if HAN_CHARACTER.search(key)}
    names, sitelinks = holders(kind)
    if kind == "persons":
        return {key for key, items in names.items() if len(items) == 1 and sitelinks[next(iter(items))] > floor}
    person_names = holders("persons")[0]
    return {
        key
        for key, items in names.items()
        if key not in person_names and any(sitelinks[item] > floor for item in items)
    }


def filtered(kind: str, spans: list[dict], text: str, names: set[str]) -> list[dict]:
    if kind == "offices":
        return office_filter(spans, text)
    if kind == "news":
        return news_filter(spans, text)
    return place_filter(spans, text, names) if kind == "places" else public_filter(spans, text, names)


def dev(kind: str, floors: list[int]) -> None:
    """Gold spans stand in for predictions, so every dropped gold person is a lost hit."""
    rows = [json.loads(line) for line in (ROOT / "data/prepared/dev.jsonl").open(encoding="utf-8")]
    for floor in floors:
        names = listed(kind, floor)
        gold = dropped = 0
        examples: Counter = Counter()
        for row in rows:
            kept = filtered(kind, row["entities"], row["text"], names)
            for span in row["entities"]:
                if span["label"] != "person":
                    continue
                gold += 1
                if span not in kept:
                    dropped += 1
                    examples[row["text"][span["start"] : span["end"]]] += 1
        print(f"floor {floor}: {len(names)} names, {dropped} of {gold} gold persons dropped ({dropped / gold:.3%})")
        print("  ", examples.most_common(15))


def wiki_predictions(system: str) -> list[list[dict]]:
    cache_file = HERE / "runs" / f"wiki_selection.{system}.jsonl"
    if cache_file.exists():
        return [json.loads(line) for line in cache_file.open(encoding="utf-8")]
    import torch

    import train_eval
    from predict import CHECKPOINTS
    from train import RUNS, load_checkpoint
    from train_eval import predict_rows

    rows = [{"text": json.loads(line)["text"], "entities": []} for line in WIKI.open(encoding="utf-8")]
    train_eval.DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, tokenizer, labels = load_checkpoint(RUNS / CHECKPOINTS[system])
    model.to(train_eval.DEVICE)
    with torch.no_grad():
        predictions = predict_rows(model, tokenizer, rows, tokenizer.pad_token_id, labels)[0]
    cache_file.write_text("".join(json.dumps(spans) + "\n" for spans in predictions))
    return predictions


def wiki(kind: str, system: str, floors: list[int]) -> None:
    texts = [json.loads(line)["text"] for line in WIKI.open(encoding="utf-8")]
    predictions = wiki_predictions(system)
    persons = sum(1 for spans in predictions for span in spans if span["label"] == "person")
    for floor in floors:
        names = listed(kind, floor)
        dropped: Counter = Counter()
        for spans, text in zip(predictions, texts, strict=True):
            kept = filtered(kind, spans, text, names)
            dropped.update(text[s["start"] : s["end"]] for s in spans if s not in kept)
        total = sum(dropped.values())
        print(f"floor {floor}: {total} of {persons} person flags dropped ({total / persons:.1%})")
        print("  ", dropped.most_common(20))


def apply(kind: str, system: str, floor: int) -> None:
    from sets import GROUPS as SET_GROUPS

    names = listed(kind, floor)
    for group in GROUPS:
        folder = HERE / "runs" if group == "comparison" else HERE / "runs" / group
        sets = SET_GROUPS[group]()
        records = [json.loads(line) for line in (folder / f"{system}.jsonl").open(encoding="utf-8")]
        dropped = 0
        with open(folder / f"{system}{SUFFIXES[kind]}.jsonl", "w", encoding="utf-8") as handle:
            for record in records:
                kept = filtered(kind, record["spans"], sets[record["set"]][record["index"]]["text"], names)
                dropped += len(record["spans"]) - len(kept)
                handle.write(json.dumps({**record, "spans": kept}) + "\n")
        print(group, len(records), "rows,", dropped, "person spans dropped")


if __name__ == "__main__":
    mode, kind = sys.argv[1], sys.argv[2]
    if kind not in SUFFIXES:
        raise SystemExit(f"KIND must be one of {sorted(SUFFIXES)}, got {kind}")
    if mode == "dev":
        dev(kind, [int(value) for value in sys.argv[3:]])
    elif mode == "wiki":
        wiki(kind, sys.argv[3], [int(value) for value in sys.argv[4:]])
    else:
        apply(kind, sys.argv[3], int(sys.argv[4]))
