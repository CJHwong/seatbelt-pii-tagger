# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "ai-edge-litert>=2.0", "huggingface_hub>=0.23,<2", "onnxruntime>=1.17", "tokenizers>=0.15",
#     "numpy", "torch>=2.2,<3", "transformers>=4.40,<6", "sentencepiece", "protobuf", "pyarrow", "pyyaml",
#     "gliner",
# ]
# ///
"""Run one system over every comparison set and write its spans to runs/SYSTEM.jsonl.

usage: HF_HUB_OFFLINE=1 uv run --script predict.py SYSTEM [GROUP]
SYSTEM is a run under first_run/runs (release), its export (release_onnx8 or release_onnx), or a
hook system: opf, redact_model, redact_hook, rules, gliner.
GROUP is comparison (the default, to runs/SYSTEM.jsonl), generalization or blind
(to runs/GROUP/SYSTEM.jsonl).
The hook systems (rules, opf, redact_*) load in this process from the agent-seatbelt clone
at data/external/agent-seatbelt, pinned at commit a830e356 (see the README). The live hook
server is never called, so the user's hook keeps its single inference slot.
"""

import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).parent
FIRST_RUN = HERE.parent / "first_run"
HOOKS = HERE.parents[1] / "data" / "external" / "agent-seatbelt" / "hooks-pii"
sys.path[:0] = [str(FIRST_RUN), str(HERE.parent / "vocab_prune"), str(HERE.parent.parent / "scripts"), str(HOOKS)]

from compare_lib import local_systems
from prune_lib import chunk_text, shifted
from sets import GROUPS
from train import RUNS, final_checkpoint, load_checkpoint, predict_rows

CHUNK_CHARS = 400
HALF_SUFFIX = "_fp16"
PERSON_BIAS = re.compile(r"^(.*)_pb([0-9.]+)$")  # NAME_pb1.5: add 1.5 to the person tags' log probabilities
CALIBRATED_SUFFIX = "_pooled"


class Checkpoints(dict):
    """Run name to checkpoint path under first_run/runs. A run named NAME is found by
    train.final_checkpoint; local_systems() lists any run stored another way."""

    def __missing__(self, name: str) -> str:
        return str(final_checkpoint(RUNS / name).relative_to(RUNS))

    def __contains__(self, name: object) -> bool:
        return super().__contains__(name) or (isinstance(name, str) and final_checkpoint(RUNS / name).exists())


CHECKPOINTS = Checkpoints(local_systems().get("checkpoints", {}))

# Systems that average the token probabilities of several runs on one build, each run
# keeping its own weights.
ENSEMBLES: dict[str, list[str]] = local_systems().get("ensembles", {})


class Ensemble(torch.nn.Module):
    """Several token classifiers as one: logits are the log of their mean probabilities."""

    def __init__(self, members: list) -> None:
        super().__init__()
        self.members = torch.nn.ModuleList(members)

    def forward(self, **inputs) -> SimpleNamespace:
        probabilities = torch.stack([member(**inputs).logits.float().softmax(-1) for member in self.members])
        return SimpleNamespace(logits=probabilities.mean(0).log())


class PersonBias(torch.nn.Module):
    """A fixed log-space bonus on every person tag before the decoder: names the model
    scores just under the other tags come out, at the price of more false names."""

    def __init__(self, model, labels: list[str], bias: float) -> None:
        super().__init__()
        self.model = model
        self.bias = torch.tensor([bias if label.endswith("-person") else 0.0 for label in labels])

    def forward(self, **inputs) -> SimpleNamespace:
        logits = self.model(**inputs).logits.float()
        return SimpleNamespace(logits=logits + self.bias.to(logits.device))


class ToolOffsets(torch.nn.Module):
    """Per-label offsets for tool call and tool output windows (segment id 1) of a run trained
    with --frame, on top of the offsets already in the classifier bias. A larger offset flags
    less, as on the outside tag. calibrate_per_frame.py fits them on unlabelled windows."""

    def __init__(self, model, labels: list[str], offsets: dict[str, float], mask_only: bool = False) -> None:
        super().__init__()
        self.model, self.mask_only = model, mask_only
        self.delta = torch.tensor(
            [-offsets.get(label.split("-", 1)[-1], 0.0) if label != "O" else 0.0 for label in labels]
        )

    def forward(self, **inputs) -> SimpleNamespace:
        types = inputs.get("token_type_ids")
        if self.mask_only and types is not None:  # a run trained without --frame reads segment 0 only
            inputs = {**inputs, "token_type_ids": torch.zeros_like(types)}
        logits = self.model(**inputs).logits.float()
        if types is None:
            return SimpleNamespace(logits=logits)
        return SimpleNamespace(logits=logits + (types == 1).unsqueeze(-1).float() * self.delta.to(logits.device))


ONNX_SUFFIXES = {"_onnx8": "model.int8.onnx", "_onnx": "model.onnx"}  # export_onnx.py output


def onnx_path(system: str) -> Path | None:
    """The export a NAME_onnx8 or NAME_onnx system scores, or None for any other name."""
    listed = local_systems().get("onnx", {})
    if system in listed:
        return HERE.parent / "export" / "build" / listed[system]
    for suffix, file in ONNX_SUFFIXES.items():
        if system.endswith(suffix):
            return HERE.parent / "export" / "build" / system.removesuffix(suffix) / file
    return None


class OnnxTagger:
    """An exported tagger behind the torch call predict_rows makes, on onnxruntime's CPU."""

    def __init__(self, path: Path) -> None:
        import onnxruntime

        self.session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def eval(self) -> "OnnxTagger":
        return self

    def __call__(self, input_ids, attention_mask) -> SimpleNamespace:
        """One row per session call, without its padding, as the hook runs a window.

        Dynamic int8 quantizes the activations of a whole call together, so in a batch a
        row's logits would depend on the rows beside it.
        """
        ids, mask = input_ids.cpu().numpy(), attention_mask.cpu().numpy()
        rows = []
        for row, length in enumerate(mask.sum(axis=1)):
            feeds = {"input_ids": ids[row : row + 1, :length], "attention_mask": mask[row : row + 1, :length]}
            logits = self.session.run(["logits"], feeds)[0][0]
            rows.append(np.pad(logits, ((0, ids.shape[1] - length), (0, 0))))
        return SimpleNamespace(logits=torch.from_numpy(np.stack(rows)))


def load_onnx(system: str):
    from transformers import PreTrainedTokenizerFast

    path = onnx_path(system)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(path.parent)
    return OnnxTagger(path), tokenizer, json.loads((path.parent / "labels.json").read_text())


def person_bias_of(system: str) -> tuple[str, float]:
    match = PERSON_BIAS.match(system)
    return (match.group(1), float(match.group(2))) if match else (system, 0.0)


def load_system(checkpoint: str):
    """One run, or the ensemble of runs named in ENSEMBLES."""
    if checkpoint not in ENSEMBLES:
        return load_checkpoint(RUNS / CHECKPOINTS[checkpoint])
    loaded = [load_checkpoint(RUNS / CHECKPOINTS[member]) for member in ENSEMBLES[checkpoint]]
    if len({tuple(labels) for _, _, labels in loaded}) != 1:
        raise SystemExit(f"{checkpoint}: the members do not share one label list")
    _, tokenizer, labels = loaded[0]
    return Ensemble([model for model, _, _ in loaded]), tokenizer, labels


def hook_predictor(system: str):
    """A text-to-spans function for one hook system, loaded in this process."""
    if system == "rules":
        from pii_rules import deterministic_spans

        return deterministic_spans
    if system == "gliner":
        from gliner_adapter import predictor

        return predictor()
    if system == "opf":
        from pii_opf import Model, ensure_assets

        return Model(ensure_assets()).predict
    import pii_redact_torch as redact

    model = redact.RedactModel(redact.ensure_assets(), "auto")
    if system == "redact_model":
        # Redact's predict merges the rules engine into its spans. Blanking the rules
        # in this process scores the model alone. The hook's file is not touched.
        redact.deterministic_spans = lambda _text: []
    return model.predict


def predict_chunked(predict, text: str) -> list[dict]:
    """The same 400-character sentence chunks our model uses; the Privacy Filter stops at 1,024 tokens."""
    spans = []
    for offset, chunk in chunk_text(text, CHUNK_CHARS):
        found = [{"start": s["start"], "end": s["end"], "label": s["label"]} for s in predict(chunk)]
        spans += shifted(found, offset)
    return spans


TOOL_KINDS = {"tool_call", "tool_output"}


def framed(rows: list[dict]) -> list[dict]:
    """For a run trained with --frame: a transcript window from a tool call or tool output gets
    segment id 1. A row with no kind (every public set) stays 0."""
    return [{**row, "frame": 1} if row.get("kind") in TOOL_KINDS else row for row in rows]


def set_predictor(system: str):
    """A rows-to-spans function, with the system loaded once. A HALF_SUFFIX name loads the
    same checkpoint with fp16 weights, which halves memory on MPS. A CALIBRATED_SUFFIX name
    folds the checkpoint's pooled maps (runs/calibration.CHECKPOINT.pooled.json) into the
    decoder, so its scores are probabilities."""
    system, bias = person_bias_of(system)
    half = system.endswith(HALF_SUFFIX)
    checkpoint = system.removesuffix(HALF_SUFFIX)
    calibrated = checkpoint.endswith(CALIBRATED_SUFFIX)
    checkpoint = checkpoint.removesuffix(CALIBRATED_SUFFIX)
    if checkpoint in CHECKPOINTS or checkpoint in ENSEMBLES:
        model, tokenizer, labels = load_system(checkpoint)
        if half:
            model.half()
        if bias:
            model = PersonBias(model, labels, bias)
        maps = None
        if calibrated:
            maps = json.loads((HERE / "runs" / f"calibration.{checkpoint}.pooled.json").read_text())["maps"]
        config = (
            json.loads((RUNS / CHECKPOINTS[checkpoint]).parent.joinpath("run.json").read_text())
            if checkpoint in CHECKPOINTS
            else {}
        )
        if config.get("frame"):
            if config.get("tool_offsets"):
                model = ToolOffsets(model, labels, config["tool_offsets"])
            return lambda rows: predict_rows(model, tokenizer, framed(rows), tokenizer.pad_token_id, labels, maps)[0]
        if config.get("kind_offsets"):  # offsets for tool windows on a run trained without --frame
            model = ToolOffsets(model, labels, config["kind_offsets"], mask_only=True)
            return lambda rows: predict_rows(model, tokenizer, framed(rows), tokenizer.pad_token_id, labels, maps)[0]
        return lambda rows: predict_rows(model, tokenizer, rows, tokenizer.pad_token_id, labels, maps)[0]
    if onnx_path(checkpoint) is not None:
        model, tokenizer, labels = load_onnx(checkpoint)
        return lambda rows: predict_rows(model, tokenizer, rows, tokenizer.pad_token_id, labels)[0]
    predict = hook_predictor(system)
    return lambda rows: [predict_chunked(predict, row["text"]) for row in rows]


def run_dir(group: str) -> Path:
    return HERE / "runs" if group == "comparison" else HERE / "runs" / group


def main() -> None:
    system, group = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "comparison"
    started = time.monotonic()
    sets = GROUPS[group]()
    predict_set = set_predictor(system)
    out = run_dir(group) / f"{system}.jsonl"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for name, rows in sets.items():
            handle.writelines(
                json.dumps({"set": name, "index": index, "spans": spans}) + "\n"
                for index, spans in enumerate(predict_set(rows))
            )
            print(f"{system} {name}: {len(rows)} rows, {time.monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
