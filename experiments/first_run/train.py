# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers", "sentencepiece", "protobuf", "numpy", "pyarrow", "pyyaml"]
# ///
"""First real training run: 18 types, pruned start, the prepared training set.

usage: uv run --script train.py train NAME STEPS [--lr LR] [--build BUILD] [--batch BATCH] [--bf16]
       uv run --script train.py eval NAME CHECKPOINT
Reads data/prepared/train.jsonl and dev.jsonl, the synthetic held-out set, and the
tokenizer and starting weights in ../vocab_prune/BUILD. Writes under runs/NAME/:
  run.json            the build the run uses; eval and the analysis scripts read it
  model-stepN.pt      weights every CHECKPOINT_EVERY steps and at the end
  loss.jsonl          loss every 50 steps
  result.json         settings, speed, and every evaluation
  eval-CHECKPOINT.json  eval mode: the evaluation of a saved checkpoint, rescored
"""

import argparse
import ast
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

import torch
import yaml
from pyarrow import parquet
from transformers import BertConfig, BertForTokenClassification, PreTrainedTokenizerFast

from augment import LONE_TOOLS, PLAIN_SWAP_WORDS, SWAP_WORDS, WIDE_LONE_TOOLS, augmented, lone_named, swapped, tool_lined
from ema import WeightAverage
from run_lib import (
    bench_gold,
    drop_name_repeats,
    drop_overlapping,
    length_batches,
    mask_ignored,
    per_type_counts,
    script_bucket,
)

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT / "experiments" / "vocab_prune"))
from build_training_set import process
from data import DATA, TW_BENCH, zh_bench_rows
from prep_lib import TYPES
from prune_lib import (
    IGNORE,
    bioes_labels,
    effective_spans,
    f1_report,
    merge_adjacent,
    overlap_counts,
    span_counts,
    token_tags,
)
from train_eval import (
    DEVICE,
    TRAIN_LENGTH,
    WARMUP_SHARE,
    learning_rate_at,
    padded,
    predict_rows,
)

HERE = Path(__file__).parent
BUILDS = HERE.parent / "vocab_prune"
RUNS = HERE / "runs"
SEED = 13
CHECKPOINT_EVERY = 5_000
PROBE_LR = 1e-3
DISTILL_SHARE = 0.5  # weight of the teacher term; the gold loss gets the rest
DISTILL_TEMPERATURE = 2.0
DISTILL_TEACHER_STEP = 10_000
POOL_BATCHES = 50  # batches per length-sorted pool
WISMUT_TEST_ROWS_PER_SCRIPT = 1_500
BENCHMARK = PROJECT / "benchmark" / "benchmark.jsonl"
# Share of each batch per script. Latin has 828k rows, Simplified 55k and Traditional 47k,
# so drawing rows uniformly would give Traditional 5% of the updates.
SCRIPT_SHARE = {"latin": 0.5, "zh-Hans": 0.3, "zh-Hant": 0.2}


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def capped(rows: list[dict], caps: list[str]) -> list[dict]:
    """Keep at most ROWS rows of SOURCE in SCRIPT, for each SOURCE:SCRIPT:ROWS, drawn with the fixed SEED."""
    for cap in caps:
        source, script, limit = cap.rsplit(":", 2)
        hit = [row for row in rows if row.get("source") == source and row["script"] == script]
        if not hit:
            raise SystemExit(f"--cap {cap}: no rows of {source} in {script}")
        kept = random.Random(SEED).sample(hit, min(len(hit), int(limit)))
        dropped = {id(row) for row in hit} - {id(row) for row in kept}
        rows = [row for row in rows if id(row) not in dropped]
    return rows


def build_of(run_dir: Path) -> Path:
    return BUILDS / json.loads((run_dir / "run.json").read_text())["build"]


def load_tokenizer(build: Path) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast.from_pretrained(build / "tokenizer-pruned")


def layer_count(state: dict) -> int:
    """Encoder layers in a BertModel state, so a build decides its own depth."""
    return 1 + max(int(key.split(".")[2]) for key in state if key.startswith("encoder.layer."))


def load_model(build: Path, vocab_size: int, pad_id: int, label_count: int) -> BertForTokenClassification:
    state = torch.load(build / "init-pruned.pt", map_location="cpu")
    rows = state["embeddings.word_embeddings.weight"].shape[0]
    if rows != vocab_size:
        raise SystemExit(f"checkpoint has {rows} embedding rows for a {vocab_size}-id tokenizer")
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=384,
        num_hidden_layers=layer_count(state),
        num_attention_heads=12,
        intermediate_size=1536,
        max_position_embeddings=512,
        type_vocab_size=2,
        pad_token_id=pad_id,
        num_labels=label_count,
    )
    model = BertForTokenClassification(config)
    state = {key: value for key, value in state.items() if not key.startswith("pooler.")}
    missing, unexpected = model.bert.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint mismatch: missing {missing}, unexpected {unexpected}")
    return model.to(DEVICE)


def encode_batch(tokenizer, rows: list[dict], label_index: dict[str, int]) -> list[dict]:
    """Encoded per batch, not up front: 930k rows of token lists do not fit in memory well."""
    encoded = tokenizer(
        [row["text"] for row in rows], truncation=True, max_length=TRAIN_LENGTH, return_offsets_mapping=True
    )
    items = []
    for row, ids, offsets in zip(rows, encoded["input_ids"], encoded["offset_mapping"], strict=True):
        spans = effective_spans(tokenizer.convert_ids_to_tokens(ids), offsets)
        tags = mask_ignored(token_tags(spans, row["entities"]), spans, row["ignore"])
        items.append({"ids": ids, "targets": [IGNORE if tag is None else label_index[tag] for tag in tags]})
    return items


def boosted_picker(rows: list[dict], source_share: dict[str, float], picker: random.Random):
    """A function that draws one row from a bucket. With source_share, each listed source takes
    its share of the bucket's draws, whatever its size. Without it, the draw is the plain
    uniform choice, so a run without the option draws the same rows as before."""
    if not source_share:
        return lambda: picker.choice(rows)
    boosted = {source: [row for row in rows if row.get("source") == source] for source in source_share}
    boosted = {source: hit for source, hit in boosted.items() if hit}
    rest = [row for row in rows if row.get("source") not in boosted]

    def draw() -> dict:
        roll = picker.random()
        for source, hit in boosted.items():
            if roll < source_share[source]:
                return picker.choice(hit)
            roll -= source_share[source]
        return picker.choice(rest)

    return draw


def batch_stream(
    tokenizer,
    rows: list[dict],
    labels: list[str],
    batch_size: int,
    seed: int = SEED,
    augment: float = 0.0,
    source_share: dict[str, float] | None = None,
    swap: float = 0.0,
    lone: float = 0.0,
    swap_words: list[str] = SWAP_WORDS,
    tools: float = 0.0,
    tool_words: list[str] = LONE_TOOLS,
) -> Iterator[list[dict]]:
    """Endless encoded batches. Rows are drawn by SCRIPT_SHARE into a pool, and the pool
    is cut into batches of similar length, so a batch carries little padding. With
    augment, that share of rows gets social-media noise from its own random stream, so
    the rows drawn stay the same as without it."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[script_bucket(row["script"])].append(row)
    names = list(SCRIPT_SHARE)
    weights = [SCRIPT_SHARE[name] for name in names]
    label_index = {label: index for index, label in enumerate(labels)}
    picker = random.Random(seed)
    noise = random.Random(seed + 1)
    swaps = random.Random(seed + 3)
    lones = random.Random(seed + 4)
    tool_lines = random.Random(seed + 5)
    draws = {name: boosted_picker(buckets[name], source_share or {}, picker) for name in names}
    while True:
        pool_rows = [draws[name]() for name in picker.choices(names, weights, k=batch_size * POOL_BATCHES)]
        pool_rows = [augmented(row, noise, augment) for row in pool_rows] if augment else pool_rows
        pool_rows = [swapped(row, swaps, swap, swap_words) for row in pool_rows] if swap else pool_rows
        pool_rows = [lone_named(row, lones, lone) for row in pool_rows] if lone else pool_rows
        pool_rows = [tool_lined(row, tool_lines, tools, tool_words) for row in pool_rows] if tools else pool_rows
        items = encode_batch(tokenizer, pool_rows, label_index)
        for indices in length_batches([len(item["ids"]) for item in items], batch_size, picker):
            yield [items[index] for index in indices]


def swap_words_of(settings) -> list[str]:
    return PLAIN_SWAP_WORDS if settings.swap_plain else SWAP_WORDS


def tool_words_of(settings) -> list[str]:
    return WIDE_LONE_TOOLS if settings.lone_tools_wide else LONE_TOOLS


def parse_share(value: str) -> tuple[str, float]:
    source, share = value.rsplit(":", 1)
    if not 0 < float(share) < 1:
        raise SystemExit(f"--source-share {value}: the share must be between 0 and 1")
    return source, float(share)


def probe_head(model, stream: Iterator[list[dict]], pad_id: int, steps: int, bf16: bool) -> None:
    """Train only the classifier on the frozen body before fine-tuning (LP-FT, Kumar et al.
    2022). A random head sends large, wrong gradients into the body early; a fitted head
    does not. The body runs in eval mode: MPS attention has no dropout without gradients."""
    model.bert.requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=PROBE_LR, weight_decay=0.01)
    for _ in range(steps):
        batch = padded(next(stream), pad_id, with_targets=True)
        with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=bf16):
            loss = model(**batch).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.bert.requires_grad_(True).train()


def load_teachers(names: list[str], build: str) -> list[BertForTokenClassification]:
    """Finished runs on the same build, frozen, whose averaged soft labels the student copies."""
    teachers = []
    for name in names:
        run_dir = RUNS / name
        if json.loads((run_dir / "run.json").read_text())["build"] != build:
            raise SystemExit(f"--teacher {name} used another build; its ids would not line up with {build}")
        teacher = load_checkpoint(run_dir / f"model-step{DISTILL_TEACHER_STEP}.pt")[0]
        teachers.append(teacher.to(DEVICE).eval().requires_grad_(False))
    return teachers


def distill_loss(student_logits: torch.Tensor, teachers: list, batch: dict) -> torch.Tensor:
    """KL from the teachers' mean tempered distribution, on tokens that carry a gold tag.
    Averaged teachers are less noisy than one run, which is the point on MPS."""
    inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
    with torch.no_grad():
        target = torch.stack(
            [(teacher(**inputs).logits.float() / DISTILL_TEMPERATURE).softmax(-1) for teacher in teachers]
        ).mean(0)
    kept = batch["labels"] != IGNORE
    student = (student_logits.float() / DISTILL_TEMPERATURE).log_softmax(-1)[kept]
    divergence = torch.nn.functional.kl_div(student, target[kept], reduction="batchmean")
    return DISTILL_TEMPERATURE**2 * divergence


def train(model, tokenizer, rows: list[dict], labels: list[str], settings: argparse.Namespace, run_dir: Path) -> dict:
    steps, lr = settings.steps, settings.lr
    shares = dict(parse_share(value) for value in settings.source_share)
    stream = batch_stream(tokenizer, rows, labels, settings.batch, settings.seed, settings.augment, shares, settings.swap_names, settings.lone_names, swap_words_of(settings), settings.lone_tools, tool_words_of(settings))
    late_step = int(steps * settings.late_from) if settings.late_share else None
    if settings.probe_steps:
        probe_head(model, stream, tokenizer.pad_token_id, settings.probe_steps, settings.bf16)
    teachers = load_teachers(settings.teacher, settings.build)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup = max(1, int(steps * WARMUP_SHARE))
    averages: list[WeightAverage] = []
    model.train()
    started = time.monotonic()
    with open(run_dir / "loss.jsonl", "w") as log:
        for step in range(steps):
            if step == late_step:  # domain upsampling at the end of training (Blakeney et al. 2024)
                late = shares | dict(parse_share(value) for value in settings.late_share)
                stream = batch_stream(
                    tokenizer, rows, labels, settings.batch, settings.seed + 2, settings.augment, late, settings.swap_names, settings.lone_names, swap_words_of(settings), settings.lone_tools, tool_words_of(settings)
                )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_at(step, steps, lr)
            batch = padded(next(stream), tokenizer.pad_token_id, with_targets=True)
            with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=settings.bf16):
                output = model(**batch)
                loss = output.loss
                if teachers:
                    loss = (1 - DISTILL_SHARE) * loss + DISTILL_SHARE * distill_loss(output.logits, teachers, batch)
            if not torch.isfinite(loss):
                raise SystemExit(f"loss is {loss.item()} at step {step}")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step >= warmup:
                averages = averages or [WeightAverage(model, decay) for decay in settings.ema]
                for average in averages:
                    average.update(model)
            if step % 50 == 0 or step == steps - 1:
                log.write(json.dumps({"step": step, "loss": loss.item(), "seconds": time.monotonic() - started}) + "\n")
                log.flush()
            if (step + 1) % CHECKPOINT_EVERY == 0 or step == steps - 1:
                torch.save(model.state_dict(), run_dir / f"model-step{step + 1}.pt")
                for average in averages:
                    torch.save(average.state_dict(model), run_dir / f"model-ema{average.decay:g}-step{step + 1}.pt")
    if DEVICE.type == "mps":
        torch.mps.synchronize()
    seconds = time.monotonic() - started
    return {"train_seconds": seconds, "steps_per_second": steps / seconds}


def score(rows: list[dict], predictions: list[list[dict]], lost: int) -> dict:
    """Typed strict overall and per type, plus the label-agnostic scores the bench comparisons use."""
    typed, untyped, lenient = Counter(), Counter(), Counter()
    by_type: dict[str, Counter] = defaultdict(Counter)
    for row, predicted in zip(rows, predictions, strict=True):
        predicted = drop_overlapping(predicted, row.get("dont_care", []))
        typed += span_counts(row["entities"], predicted, typed=True)
        for label, counts in per_type_counts(row["entities"], predicted).items():
            by_type[label] += counts
        gold_merged = merge_adjacent(row["entities"], row["text"])
        predicted_merged = merge_adjacent(predicted, row["text"])
        untyped += span_counts(gold_merged, predicted_merged, typed=False)
        lenient += overlap_counts(gold_merged, predicted_merged)
    return {
        "rows": len(rows),
        "typed_strict": f1_report(typed),
        "untyped_merged_strict": f1_report(untyped),
        "untyped_overlap": f1_report(lenient),
        "per_type": {label: f1_report(counts) for label, counts in sorted(by_type.items())},
        "gold_spans_lost_to_truncation": lost,
    }


def tw_bench() -> list[dict]:
    rows = []
    for part in ("short", "mid", "long"):
        for row in parquet.read_table(TW_BENCH / f"{part}.parquet").to_pylist():
            spans = row["spans"] if isinstance(row["spans"], list) else ast.literal_eval(row["spans"])
            entities, dont_care = bench_gold(row["text"], spans)
            negative = str(row["is_negative"]) == "True"
            rows.append({"text": row["text"], "entities": entities, "dont_care": dont_care, "negative": negative})
    return rows


def evaluate_tw_bench(model, tokenizer, labels: list[str]) -> dict:
    rows = tw_bench()
    predictions, lost = predict_rows(model, tokenizer, rows, tokenizer.pad_token_id, labels)
    filtered = [drop_name_repeats(p, row["text"], row["entities"]) for row, p in zip(rows, predictions, strict=True)]
    predictions = [kept for kept, _ in filtered]
    report = score(rows, predictions, lost)
    negatives = [p for row, p in zip(rows, predictions, strict=True) if row["negative"]]
    flagged = sum(
        1
        for row, predicted in zip(rows, predictions, strict=True)
        for date in row["dont_care"]
        if len(drop_overlapping(predicted, [date])) < len(predicted)
    )
    report |= {
        "negative_rows": len(negatives),
        "negative_rows_flagged": sum(1 for p in negatives if p),
        "other_dates_set_aside": sum(len(row["dont_care"]) for row in rows),
        "other_dates_flagged": flagged,
        "name_repeats_set_aside": sum(count for _, count in filtered),
    }
    return report


def wismut_test() -> dict[str, list[dict]]:
    """Wismut's test split in our types. 11% of its texts also sit in its train split,
    so every row whose text is in our training or dev set is dropped first."""
    datasets = yaml.safe_load((PROJECT / "type_map.yaml").read_text())["datasets"]
    config = datasets["Wismut/nym-pii-multilingual-data"]
    seen = {
        hashlib.sha256(row["text"].encode()).digest()
        for name in ("train", "dev")
        for row in read_jsonl(DATA / f"prepared/{name}.jsonl")
    }
    by_script: dict[str, list[dict]] = defaultdict(list)
    for raw in read_jsonl(DATA / "Wismut/nym-pii-multilingual-data/test.jsonl"):
        if hashlib.sha256(raw["text"].encode()).digest() in seen:
            continue
        record, _, _ = process({"text": raw["text"], "spans": raw["entities"], "language": None}, config)
        if record:
            by_script["latin" if record["script"] == "latin" else "zh"].append(
                {**record, "dont_care": record["ignore"]}
            )
    picker = random.Random(SEED)
    return {name: picker.sample(rows, min(len(rows), WISMUT_TEST_ROWS_PER_SCRIPT)) for name, rows in by_script.items()}


def evaluate_benchmark(model, tokenizer, labels: list[str]) -> dict:
    """Our own Traditional benchmark: report only, never used to choose anything."""
    rows = [{**row, "dont_care": row["ignore"]} for row in read_jsonl(BENCHMARK)]
    predictions, lost = predict_rows(model, tokenizer, rows, tokenizer.pad_token_id, labels)
    report = score(rows, predictions, lost)
    negatives = [p for row, p in zip(rows, predictions, strict=True) if not row["entities"]]
    report |= {"negative_rows": len(negatives), "negative_rows_flagged": sum(1 for p in negatives if p)}
    return report


def evaluate(model, tokenizer, labels: list[str]) -> dict:
    started = time.monotonic()
    dev = read_jsonl(DATA / "prepared/dev.jsonl")
    heldout = [
        {"text": row["text"], "entities": [s for s in row["spans"] if s["label"] != "ignore"]}
        for row in read_jsonl(DATA / "synthetic/heldout.jsonl")
    ]
    report = {}
    for bucket in SCRIPT_SHARE:
        rows = [row for row in dev if script_bucket(row["script"]) == bucket]
        report[f"dev_{bucket}"] = score(rows, *predict_rows(model, tokenizer, rows, tokenizer.pad_token_id, labels))
    report["synthetic_heldout"] = score(
        heldout, *predict_rows(model, tokenizer, heldout, tokenizer.pad_token_id, labels)
    )
    report["tw_bench"] = evaluate_tw_bench(model, tokenizer, labels)
    zh = zh_bench_rows()
    report["zh_bench"] = score(zh, *predict_rows(model, tokenizer, zh, tokenizer.pad_token_id, labels))
    report["zh_bench"].pop("typed_strict")
    report["zh_bench"].pop("per_type")
    for name, rows in wismut_test().items():
        report[f"wismut_test_{name}"] = score(
            rows, *predict_rows(model, tokenizer, rows, tokenizer.pad_token_id, labels)
        )
    if BENCHMARK.exists():
        report["our_benchmark"] = evaluate_benchmark(model, tokenizer, labels)
    report["eval_seconds"] = time.monotonic() - started
    return report


def final_checkpoint(run_dir: Path) -> Path:
    """model-step10000.pt, the step the release recipe trains to, or the last step a shorter run saved."""
    standard = run_dir / "model-step10000.pt"
    if standard.exists():
        return standard
    saved = sorted(run_dir.glob("model-step*.pt"), key=lambda path: int(path.stem.removeprefix("model-step")))
    return saved[-1] if saved else standard


def load_checkpoint(checkpoint: Path) -> tuple[BertForTokenClassification, PreTrainedTokenizerFast, list[str]]:
    build = build_of(checkpoint.parent)
    tokenizer = load_tokenizer(build)
    labels = bioes_labels(list(TYPES))
    model = load_model(build, len(tokenizer), tokenizer.pad_token_id, len(labels))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model, tokenizer, labels


def evaluate_checkpoint(run_dir: Path, checkpoint: str) -> None:
    model, tokenizer, labels = load_checkpoint(run_dir / checkpoint)
    report = evaluate(model, tokenizer, labels)
    (run_dir / f"eval-{Path(checkpoint).stem}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(dest="mode", required=True)
    train_mode = modes.add_parser("train")
    train_mode.add_argument("name")
    train_mode.add_argument("steps", type=int)
    train_mode.add_argument("--lr", type=float, default=3e-4)
    train_mode.add_argument("--build", default="build-prepared")
    train_mode.add_argument("--batch", type=int, default=32)
    train_mode.add_argument("--bf16", action="store_true", help="bfloat16 autocast, about 38%% faster on Apple silicon")
    train_mode.add_argument(
        "--without", action="append", default=[], metavar="SOURCE", help="drop a source's rows, for an A/B control"
    )
    train_mode.add_argument(
        "--cap", action="append", default=[], metavar="SOURCE:SCRIPT:ROWS", help="keep at most ROWS of SOURCE in SCRIPT"
    )
    train_mode.add_argument(
        "--ema",
        type=float,
        action="append",
        default=[],
        metavar="DECAY",
        help="also save a moving average of the weights with this decay, from the end of warmup",
    )
    train_mode.add_argument(
        "--augment", type=float, default=0.0, metavar="SHARE", help="share of rows given social-media noise"
    )
    train_mode.add_argument(
        "--swap-names",
        type=float,
        default=0.0,
        metavar="SHARE",
        help="share of Latin rows whose person spans are filled with words that are not names",
    )
    train_mode.add_argument(
        "--lone-names",
        type=float,
        default=0.0,
        metavar="SHARE",
        help="share of developer-context rows given one line with a lone first name or handle",
    )
    train_mode.add_argument(
        "--swap-plain", action="store_true", help="--swap-names uses plain and invented words only, no tool names"
    )
    train_mode.add_argument(
        "--lone-tools",
        type=float,
        default=0.0,
        metavar="SHARE",
        help="share of developer-context rows given one unlabelled line with a name-like tool name",
    )
    train_mode.add_argument(
        "--lone-tools-wide", action="store_true", help="--lone-tools draws from the wider tool-name list"
    )
    train_mode.add_argument(
        "--source-share",
        action="append",
        default=[],
        metavar="SOURCE:SHARE",
        help="give SOURCE this share of the draws in every script bucket it has rows in",
    )
    train_mode.add_argument(
        "--late-share",
        action="append",
        default=[],
        metavar="SOURCE:SHARE",
        help="from --late-from on, give SOURCE this share instead (repeatable)",
    )
    train_mode.add_argument(
        "--late-from",
        type=float,
        default=0.85,
        metavar="FRACTION",
        help="the share of steps before --late-share applies",
    )
    train_mode.add_argument(
        "--teacher", action="append", default=[], metavar="RUN", help="distil from this finished run (repeatable)"
    )
    train_mode.add_argument(
        "--probe-steps", type=int, default=0, metavar="N", help="first train only the head for N steps (LP-FT)"
    )
    train_mode.add_argument(
        "--seed", type=int, default=SEED, help="weights and batch order only; the evaluation rows stay fixed"
    )
    eval_mode = modes.add_parser("eval")
    eval_mode.add_argument("name")
    eval_mode.add_argument("checkpoint")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    run_dir = RUNS / arguments.name
    if arguments.mode == "eval":
        evaluate_checkpoint(run_dir, arguments.checkpoint)
        return
    steps, lr = arguments.steps, arguments.lr
    torch.manual_seed(arguments.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps({"build": arguments.build}))
    build = build_of(run_dir)
    tokenizer = load_tokenizer(build)
    labels = bioes_labels(list(TYPES))
    rows = [row for row in read_jsonl(DATA / "prepared/train.jsonl") if row.get("source") not in arguments.without]
    rows = capped(rows, arguments.cap)
    model = load_model(build, len(tokenizer), tokenizer.pad_token_id, len(labels))
    result = {
        "build": arguments.build,
        "steps": steps,
        "lr": lr,
        "batch": arguments.batch,
        "bf16": arguments.bf16,
        "without": arguments.without,
        "seed": arguments.seed,
        "cap": arguments.cap,
        "ema": arguments.ema,
        "augment": arguments.augment,
        "swap_names": arguments.swap_names,
        "lone_names": arguments.lone_names,
        "swap_plain": arguments.swap_plain,
        "lone_tools": arguments.lone_tools,
        "lone_tools_wide": arguments.lone_tools_wide,
        "probe_steps": arguments.probe_steps,
        "teachers": arguments.teacher,
        "source_share": arguments.source_share,
        "late_share": arguments.late_share,
        "late_from": arguments.late_from if arguments.late_share else None,
        "length_sorted_batches": True,
        "script_share": SCRIPT_SHARE,
        "train_rows": len(rows),
        "labels": len(labels),
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    result |= train(model, tokenizer, rows, labels, arguments, run_dir)
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    result["eval"] = evaluate(model, tokenizer, labels)
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
