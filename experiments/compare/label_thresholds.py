# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers", "sentencepiece", "protobuf", "numpy", "pyarrow", "pyyaml"]
# ///
"""Choose one confidence threshold per label on the dev split, then apply it as a new system.

usage: uv run --script label_thresholds.py choose CHECKPOINT        -> runs/thresholds.CHECKPOINT.json
       uv run --script label_thresholds.py apply SYSTEM [THRESHOLDS] -> runs/[GROUP/]SYSTEM_thr.jsonl

CHECKPOINT is a predict.py checkpoint or ensemble name. apply takes THRESHOLDS, either a
file such as thresholds.json (the released cuts, which chain.sh applies) or a CHECKPOINT
whose runs/thresholds.CHECKPOINT.json choose wrote; SYSTEM itself when omitted. So the cuts
chosen on a raw checkpoint can apply to its chained output.

choose predicts data/prepared/dev.jsonl with SYSTEM's checkpoint and keeps, per label, the
threshold with the best typed overlap F1 there. The dev split comes from the training
sources, never from an evaluation set. apply drops each label's spans below its threshold
in every group's predictions, so score.py can score SYSTEM_thr like any other system.
"""

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path[:0] = [str(ROOT / "experiments/first_run"), str(ROOT / "experiments/vocab_prune")]

GRID = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
GROUPS = ["comparison", "generalization", "blind", "rules", "benchmarks", "published"]


def run_dir(group: str) -> Path:
    return HERE / "runs" if group == "comparison" else HERE / "runs" / group


def overlaps(first: dict, second: dict) -> bool:
    return first["start"] < second["end"] and second["start"] < first["end"]


def label_counts(rows: list[dict], predictions: list[list[dict]], label: str, threshold: float) -> Counter:
    counts: Counter = Counter()
    for row, predicted in zip(rows, predictions, strict=True):
        gold = [g for g in row["entities"] if g["label"] == label]
        kept = [p for p in predicted if p["label"] == label and p["score"] >= threshold]
        counts["gold"] += len(gold)
        counts["predicted"] += len(kept)
        counts["found"] += sum(any(overlaps(g, p) for p in kept) for g in gold)
        counts["correct"] += sum(any(overlaps(p, g) for g in gold) for p in kept)
    return counts


def f1(counts: Counter) -> float:
    precision = counts["correct"] / counts["predicted"] if counts["predicted"] else 0.0
    recall = counts["found"] / counts["gold"] if counts["gold"] else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def choose(system: str) -> None:
    import torch

    import train_eval
    from predict import load_system
    from train_eval import predict_rows

    rows = [json.loads(line) for line in (ROOT / "data/prepared/dev.jsonl").open(encoding="utf-8")]
    train_eval.DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, tokenizer, labels = load_system(system)
    model.to(train_eval.DEVICE)
    with torch.no_grad():
        predictions = predict_rows(model, tokenizer, rows, tokenizer.pad_token_id, labels)[0]
    chosen = {}
    for label in sorted({g["label"] for row in rows for g in row["entities"]}):
        scores = {t: f1(label_counts(rows, predictions, label, t)) for t in GRID}
        best = max(GRID, key=lambda t: (round(scores[t], 4), -t))
        chosen[label] = {"threshold": best, "dev_f1": scores[best], "dev_f1_at_0": scores[0.0]}
        print(label, chosen[label], flush=True)
    (HERE / "runs" / f"thresholds.{system}.json").write_text(json.dumps(chosen, indent=1))


def thresholds_path(name: str) -> Path:
    return HERE / name if name.endswith(".json") else HERE / "runs" / f"thresholds.{name}.json"


def apply(system: str, thresholds: str | None = None) -> None:
    chosen = json.loads(thresholds_path(thresholds or system).read_text())
    for group in GROUPS:
        records = [json.loads(line) for line in (run_dir(group) / f"{system}.jsonl").open(encoding="utf-8")]
        with open(run_dir(group) / f"{system}_thr.jsonl", "w", encoding="utf-8") as handle:
            for record in records:
                spans = [s for s in record["spans"] if s["score"] >= chosen.get(s["label"], {}).get("threshold", 0.0)]
                handle.write(json.dumps({**record, "spans": spans}) + "\n")
        print(group, len(records), "rows")


if __name__ == "__main__":
    {"choose": choose, "apply": apply}[sys.argv[1]](*sys.argv[2:])
