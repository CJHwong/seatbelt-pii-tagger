# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "ai-edge-litert>=2.0", "huggingface_hub>=0.23,<2", "onnxruntime>=1.17", "tokenizers>=0.15",
#     "numpy", "torch>=2.2,<3", "transformers>=4.40,<6", "sentencepiece", "protobuf", "pyarrow", "pyyaml",
# ]
# ///
"""Time one request at a time, as the hook sends them, and write runs/latency.json.

usage: HF_HUB_OFFLINE=1 uv run --script latency.py [TAG SYSTEM...]

With no arguments, every checkpoint and the hook systems are measured and written to
runs/latency.json. With a TAG and system names (checkpoints or predict.py ensembles),
only those are measured, on cpu and mps, and written to runs/latency.TAG.json.

Every system reads the same texts, one text per call, after a warm-up. The texts are
short Traditional rows, English rows and long documents, so the cost of length shows.
The Privacy Filter runs on CPU only, so ours and Redact run on both CPU and the GPU.
"""

import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE.parent / "first_run"), str(HERE.parent / "vocab_prune"), str(HERE.parent.parent / "scripts")]
import train_eval
from predict import CHECKPOINTS, HOOKS, load_system, predict_chunked
from sets import blind_sets, comparison_sets, generalization_sets
from train import predict_rows

sys.path.insert(0, str(HOOKS))
WARMUP = 10
SEED = 13


def texts() -> list[str]:
    generator = random.Random(SEED)
    short = [row["text"] for row in generator.sample(comparison_sets()["our_benchmark"], 150)]
    english = [row["text"] for row in generator.sample(blind_sets()["blind_en"], 100)]
    long = [row["text"] for row in generator.sample(generalization_sets()["tab_echr"], 10)]
    long += [row["text"] for row in generalization_sets()["perturbed_long"]]
    return short + english + long


def ours(checkpoint: str, device: str):
    train_eval.DEVICE = torch.device(device)
    model, tokenizer, labels = load_system(checkpoint)
    model.to(train_eval.DEVICE)
    predict = lambda text: predict_rows(
        model, tokenizer, [{"text": text, "entities": []}], tokenizer.pad_token_id, labels
    )[0][0]
    if device == "mps":
        return lambda text: (predict(text), torch.mps.synchronize())
    return predict


def hook(system: str, device: str):
    if system == "rules":
        from pii_rules import deterministic_spans

        return lambda text: predict_chunked(deterministic_spans, text)
    if system == "opf":
        from pii_opf import Model, ensure_assets

        model = Model(ensure_assets())
        return lambda text: predict_chunked(model.predict, text)
    import pii_redact_torch as redact

    model = redact.RedactModel(redact.ensure_assets(), device)
    return lambda text: predict_chunked(model.predict, text)


def measure(predict, inputs: list[str]) -> dict:
    for text in inputs[:WARMUP]:
        predict(text)
    seconds = []
    for text in inputs:
        started = time.perf_counter()
        predict(text)
        seconds.append(time.perf_counter() - started)
    milliseconds = sorted(1000 * s for s in seconds)
    characters = sum(map(len, inputs))
    return {
        "mean_ms": statistics.fmean(milliseconds),
        "p50_ms": milliseconds[len(milliseconds) // 2],
        "p95_ms": milliseconds[int(len(milliseconds) * 0.95)],
        "max_ms": milliseconds[-1],
        "ms_per_1k_chars": 1000 * sum(seconds) * 1000 / characters,
    }


def main() -> None:
    inputs = texts()
    lengths = sorted(map(len, inputs))
    print(f"{len(inputs)} texts, median {lengths[len(lengths) // 2]} chars, max {lengths[-1]}", flush=True)
    chosen = sys.argv[2:]
    names = chosen or list(CHECKPOINTS)
    runs = [(f"{name} {device}", lambda n=name, d=device: ours(n, d)) for name in names for device in ("cpu", "mps")]
    if not chosen:
        runs += [("opf cpu", lambda: hook("opf", "cpu"))]
        runs += [(f"redact_hook {device}", lambda d=device: hook("redact_hook", d)) for device in ("cpu", "mps")]
        runs += [("rules cpu", lambda: hook("rules", "cpu"))]
    results = {}
    for name, build in runs:
        results[name] = measure(build(), inputs)
        print(name, {k: round(v, 1) for k, v in results[name].items()}, flush=True)
    out = HERE / "runs" / (f"latency.{sys.argv[1]}.json" if chosen else "latency.json")
    out.write_text(json.dumps({"texts": len(inputs), "results": results}, indent=1))


if __name__ == "__main__":
    main()
