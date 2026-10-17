# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers", "sentencepiece", "protobuf", "numpy", "pyarrow"]
# ///
"""Train one arm of the pruning experiment for a fixed step budget, then evaluate it.

usage: uv run train_eval.py ARM LR STEPS
  ARM is pruned (MiniLM rows for the 64,000 kept pieces), random (same shape and
  tokenizer, random weights) or full (MiniLM with all 250,037 rows, the prune's control).
Writes runs/ARM-lrLR.json and runs/ARM-lrLR.loss.jsonl.
"""

import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy
import torch
from transformers import BertConfig, BertForTokenClassification, PreTrainedTokenizerFast

from data import (
    entity_types,
    training_rows,
    tw_bench_rows,
    wismut_eval_rows,
    zh_bench_rows,
)
from prune_lib import (
    IGNORE,
    bioes_labels,
    boundary_misaligned,
    calibrated_spans,
    chunk_text,
    decode_spans,
    effective_spans,
    f1_report,
    join_touching,
    merge_adjacent,
    overlap_counts,
    shifted,
    span_counts,
    token_tags,
    transition_matrix,
    viterbi_numpy,
)

HERE = Path(__file__).parent
BUILD = HERE / "build"
RUNS = HERE / "runs"
ARMS = {
    "pruned": ("tokenizer-pruned", "init-pruned.pt"),
    "random": ("tokenizer-pruned", None),
    "full": ("tokenizer-full", "init-full.pt"),
}
TRAIN_LENGTH = 256
EVAL_LENGTH = 512
CHUNK_CHARS = 400
BATCH = 32
EVAL_BATCH = 64
WARMUP_SHARE = 0.06
SEED = 13
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def encode(tokenizer, rows: list[dict], label_index: dict[str, int] | None, length: int) -> list[dict]:
    """Token ids, BIOES targets and character spans per row. label_index None means no targets."""
    encoded = tokenizer([row["text"] for row in rows], truncation=True, max_length=length, return_offsets_mapping=True)
    items = []
    for row, ids, offsets in zip(rows, encoded["input_ids"], encoded["offset_mapping"], strict=True):
        spans = effective_spans(tokenizer.convert_ids_to_tokens(ids), offsets)
        item = {"ids": ids, "spans": spans, "truncated_at": max((s[1] for s in spans if s), default=0)}
        if row.get("frame"):  # 1 for a tool call or tool output window; read as the segment id
            item["frame"] = row["frame"]
        if label_index is not None:
            tags = token_tags(spans, row["entities"])
            item["targets"] = [IGNORE if tag is None else label_index[tag] for tag in tags]
        items.append(item)
    return items


def alignment_loss(items: list[dict], rows: list[dict]) -> float:
    entities = [(item, entity) for item, row in zip(items, rows, strict=True) for entity in row["entities"]]
    visible = [(item, entity) for item, entity in entities if entity["end"] <= item["truncated_at"]]
    lost = sum(1 for item, entity in visible if boundary_misaligned(item["spans"], entity))
    return lost / len(visible) if visible else 0.0


def build_model(arm: str, vocab_size: int, pad_id: int, label_count: int) -> BertForTokenClassification:
    init_file = ARMS[arm][1]
    state = torch.load(BUILD / init_file, map_location="cpu") if init_file else None
    if state is not None:
        # MiniLM ships 250,037 embedding rows for a 250,002-id tokenizer. The 35 extra
        # rows are unreachable padding, so the checkpoint, not the tokenizer, sets the size.
        rows = state["embeddings.word_embeddings.weight"].shape[0]
        if rows < vocab_size:
            raise SystemExit(f"checkpoint has {rows} embedding rows for a {vocab_size}-id tokenizer")
        vocab_size = rows
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=1536,
        max_position_embeddings=512,
        type_vocab_size=2,
        pad_token_id=pad_id,
        num_labels=label_count,
    )
    model = BertForTokenClassification(config)
    if state is not None:
        state = {key: value for key, value in state.items() if not key.startswith("pooler.")}
        missing, unexpected = model.bert.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise SystemExit(f"checkpoint mismatch: missing {missing}, unexpected {unexpected}")
    return model.to(DEVICE)


def padded(items: list[dict], pad_id: int, with_targets: bool) -> dict[str, torch.Tensor]:
    width = max(len(item["ids"]) for item in items)
    ids = torch.full((len(items), width), pad_id)
    mask = torch.zeros((len(items), width), dtype=torch.long)
    targets = torch.full((len(items), width), IGNORE)
    for row, item in enumerate(items):
        ids[row, : len(item["ids"])] = torch.tensor(item["ids"])
        mask[row, : len(item["ids"])] = 1
        if with_targets:
            targets[row, : len(item["ids"])] = torch.tensor(item["targets"])
    batch = {"input_ids": ids.to(DEVICE), "attention_mask": mask.to(DEVICE)}
    if any(item.get("frame") for item in items):  # only frame-trained runs get rows with a frame
        types = torch.zeros((len(items), width), dtype=torch.long)
        for row, item in enumerate(items):
            types[row, : len(item["ids"])] = item.get("frame", 0)
        batch["token_type_ids"] = types.to(DEVICE)
    if with_targets:
        batch["labels"] = targets.to(DEVICE)
    return batch


def learning_rate_at(step: int, steps: int, peak: float) -> float:
    warmup = max(1, int(steps * WARMUP_SHARE))
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * max(0.0, (steps - step) / (steps - warmup))


def train(model, items: list[dict], pad_id: int, lr: float, steps: int, loss_log: Path) -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    order = random.Random(SEED)
    queue: list[int] = []
    model.train()
    started = time.monotonic()
    with open(loss_log, "w") as log:
        for step in range(steps):
            if len(queue) < BATCH:
                queue += order.sample(range(len(items)), len(items))
            batch_items, queue = [items[i] for i in queue[:BATCH]], queue[BATCH:]
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_at(step, steps, lr)
            loss = model(**padded(batch_items, pad_id, with_targets=True)).loss
            if not torch.isfinite(loss):
                raise SystemExit(f"loss is {loss.item()} at step {step}")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % 50 == 0 or step == steps - 1:
                log.write(json.dumps({"step": step, "loss": loss.item(), "seconds": time.monotonic() - started}) + "\n")
                log.flush()
    if DEVICE.type == "mps":
        torch.mps.synchronize()
    seconds = time.monotonic() - started
    return {"train_seconds": seconds, "steps_per_second": steps / seconds}


@torch.no_grad()
def predict(model, items: list[dict], pad_id: int, labels: list[str]) -> list[list[dict]]:
    model.eval()
    transitions = transition_matrix(labels)
    predictions = []
    for start in range(0, len(items), EVAL_BATCH):
        chunk = items[start : start + EVAL_BATCH]
        logits = model(**padded(chunk, pad_id, with_targets=False)).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
        predictions += [
            _decode_one(item, rows, labels, transitions) for item, rows in zip(chunk, log_probs, strict=False)
        ]
    return predictions


def _decode_one(item: dict, log_probs: numpy.ndarray, labels: list[str], transitions) -> list[dict]:
    real = [index for index, span in enumerate(item["spans"]) if span]
    if not real:
        return []
    path = viterbi_numpy(log_probs[real], labels, transitions)
    confidence = numpy.exp(log_probs[real][numpy.arange(len(path)), path]).tolist()
    return decode_spans([labels[index] for index in path], [item["spans"][index] for index in real], confidence)


def predict_rows(
    model, tokenizer, rows: list[dict], pad_id: int, labels: list[str], calibration: dict | None = None
) -> tuple[list[list[dict]], int]:
    """Predict whole rows by chunking at sentence ends. Returns predictions and gold spans no chunk saw.

    calibration holds Platt maps from compare/calibrate.py. They apply to the final spans,
    after chunks are joined, because that is where they were fitted."""
    pieces = [
        (row_index, offset, chunk)
        for row_index, row in enumerate(rows)
        for offset, chunk in chunk_text(row["text"], CHUNK_CHARS)
    ]
    items = encode(
        tokenizer,
        [{"text": chunk, "frame": rows[row_index].get("frame")} for row_index, _, chunk in pieces],
        None,
        EVAL_LENGTH,
    )
    chunk_predictions = predict(model, items, pad_id, labels)
    predictions: list[list[dict]] = [[] for _ in rows]
    visible: list[list[tuple[int, int]]] = [[] for _ in rows]
    for (row_index, offset, _), item, predicted in zip(pieces, items, chunk_predictions, strict=True):
        predictions[row_index] += shifted(predicted, offset)
        visible[row_index].append((offset, offset + item["truncated_at"]))
    lost = sum(
        1
        for row, windows in zip(rows, visible, strict=True)
        for entity in row["entities"]
        if not any(start <= entity["start"] and entity["end"] <= end for start, end in windows)
    )
    return [calibrated_spans(join_touching(spans), calibration) for spans in predictions], lost


def score(rows: list[dict], predictions: list[list[dict]], lost_to_truncation: int) -> dict:
    typed, untyped, lenient = Counter(), Counter(), Counter()
    for row, predicted in zip(rows, predictions, strict=True):
        typed += span_counts(row["entities"], predicted, typed=True)
        gold_merged = merge_adjacent(row["entities"], row["text"])
        predicted_merged = merge_adjacent(predicted, row["text"])
        untyped += span_counts(gold_merged, predicted_merged, typed=False)
        lenient += overlap_counts(gold_merged, predicted_merged)
    return {
        "rows": len(rows),
        "typed_strict": f1_report(typed),
        "untyped_merged_strict": f1_report(untyped),
        "untyped_overlap": f1_report(lenient),
        "gold_spans_lost_to_truncation": lost_to_truncation,
    }


def evaluate(model, tokenizer, pad_id: int, labels: list[str]) -> dict:
    started = time.monotonic()
    wismut = wismut_eval_rows()
    tw_rows = tw_bench_rows()
    sets = {
        "wismut_test_latin": [row for row in wismut if row["script"] == "latin"],
        "wismut_test_zh": [row for row in wismut if row["script"] == "zh"],
        "tw_bench": tw_rows,
        "zh_bench": zh_bench_rows(),
    }
    report = {}
    for name, rows in sets.items():
        predictions, lost = predict_rows(model, tokenizer, rows, pad_id, labels)
        report[name] = score(rows, predictions, lost)
        if name == "tw_bench":
            negatives = [p for row, p in zip(rows, predictions, strict=True) if row["negative"]]
            report[name]["negative_rows"] = len(negatives)
            report[name]["negative_rows_flagged"] = sum(1 for p in negatives if p)
    report["eval_seconds"] = time.monotonic() - started
    return report


def main() -> None:
    arm, lr, steps = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    torch.manual_seed(SEED)
    RUNS.mkdir(exist_ok=True)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(BUILD / ARMS[arm][0])
    labels = bioes_labels(entity_types())
    rows = training_rows()
    items = encode(tokenizer, rows, {label: index for index, label in enumerate(labels)}, TRAIN_LENGTH)
    model = build_model(arm, len(tokenizer), tokenizer.pad_token_id, len(labels))
    name = f"{arm}-lr{lr:g}"
    result = {
        "arm": arm,
        "lr": lr,
        "steps": steps,
        "batch": BATCH,
        "train_rows": len(rows),
        "labels": len(labels),
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_entity_boundary_misaligned": alignment_loss(items, rows),
    }
    result |= train(model, items, tokenizer.pad_token_id, lr, steps, RUNS / f"{name}.loss.jsonl")
    result["eval"] = evaluate(model, tokenizer, tokenizer.pad_token_id, labels)
    (RUNS / f"{name}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
