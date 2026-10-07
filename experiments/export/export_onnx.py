# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch>=2.2,<3", "transformers>=4.40,<6", "tokenizers>=0.15", "sentencepiece", "protobuf",
#     "numpy", "pyyaml", "pyarrow", "onnx", "onnxscript", "onnxruntime>=1.17",
# ]
# ///
"""Export one checkpoint to the files the agent-seatbelt hook loads.

usage: uv run --script export_onnx.py [RUN]   RUN defaults to release, the run reproduce.sh trains

Writes build/RUN/: model.onnx (fp32), model.int8.onnx (dynamic int8 weights), both with
the attention, layer norm and gelu ops fused by onnxruntime's BERT optimizer,
the tokenizer files, labels.json, persons.txt.gz and places.txt.gz (the public names
the chain drops, at floor 20) and tagger.json (window sizes and the url and username cuts,
as chain.sh applies them). The int8 file ships only if it scores inside the noise
floor of the fp32 one (predict.py systems RUN_onnx and RUN_onnx8).
"""

import gzip
import json
import math
import shutil
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE.parent / p) for p in ("first_run", "vocab_prune", "compare")] + [
    str(HERE.parent.parent / "scripts")
]

from train import RUNS, build_of, final_checkpoint, load_checkpoint
from train_eval import CHUNK_CHARS, EVAL_LENGTH

FLOORS = {"persons": 20, "places": 20}  # chain.sh
THRESHOLDS = HERE.parent / "compare" / "thresholds.json"  # chain.sh applies the same cuts


class LogitsOnly(torch.nn.Module):
    """The tagger with a plain tensor output, so the graph has one named output.

    It passes the additive mask BERT always used, not the mask transformers 5 builds,
    because onnxruntime's optimizer fuses attention only on that pattern.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        additive = (1.0 - attention_mask[:, None, None, :].to(torch.float32)) * torch.finfo(torch.float32).min
        return self.model(input_ids=input_ids, attention_mask=additive).logits


def classic_attention(self, hidden_states, attention_mask=None, **kwargs):
    """BERT self-attention as transformers 4 traced it, the graph the optimizer fuses.

    The same arithmetic as transformers 5's eager attention: the fused graph's logits
    differ from the plain export by at most 3e-05.
    """

    def heads(layer):
        shape = layer.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return layer.view(shape).permute(0, 2, 1, 3)

    query, key, value = (
        heads(self.query(hidden_states)),
        heads(self.key(hidden_states)),
        heads(self.value(hidden_states)),
    )
    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attention_head_size) + attention_mask
    context = torch.matmul(torch.softmax(scores, dim=-1), value).permute(0, 2, 1, 3).contiguous()
    return context.view(context.size()[:-2] + (self.all_head_size,)), None


def export(run: str) -> Path:
    out = HERE / "build" / run
    out.mkdir(parents=True, exist_ok=True)
    model, _, labels = load_checkpoint(final_checkpoint(RUNS / run))
    model.to("cpu").eval()
    from transformers.models.bert import modeling_bert

    modeling_bert.BertSelfAttention.forward = classic_attention
    ids = torch.ones((2, 64), dtype=torch.long)
    axes = {"b": 0, "t": 1}
    torch.onnx.export(
        LogitsOnly(model),
        (ids, torch.ones_like(ids)),
        str(out / "model.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            name: {index: axis for axis, index in axes.items()} for name in ("input_ids", "attention_mask", "logits")
        },
        opset_version=17,
        dynamo=False,
    )
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.transformers import optimizer

    fused = optimizer.optimize_model(
        str(out / "model.onnx"),
        model_type="bert",
        num_heads=model.config.num_attention_heads,
        hidden_size=model.config.hidden_size,
        opt_level=0,
    )
    counts = fused.get_fused_operator_statistics()
    if counts["Attention"] != model.config.num_hidden_layers:
        raise SystemExit(f"fused {counts['Attention']} attention ops for {model.config.num_hidden_layers} layers")
    fused.save_model_to_file(str(out / "model.onnx"))
    # The fused ops come from onnxruntime's own domain, where shape inference finds no type.
    quantize_dynamic(
        str(out / "model.onnx"),
        str(out / "model.int8.onnx"),
        weight_type=QuantType.QInt8,
        extra_options={"DefaultTensorType": 1},
    )
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(build_of(RUNS / run) / "tokenizer-pruned" / name, out / name)
    (out / "labels.json").write_text(json.dumps(labels))
    for name in ("model.onnx", "model.int8.onnx", "tokenizer.json"):
        print(name, round((out / name).stat().st_size / 1e6, 1), "MB")
    export_chain(out)
    return out


def export_chain(out: Path) -> None:
    from public_filter import listed

    for kind, floor in FLOORS.items():
        names = sorted(listed(kind, floor))
        with gzip.open(out / f"{kind}.txt.gz", "wt", encoding="utf-8") as handle:
            handle.writelines(name + "\n" for name in names)
        print(kind, len(names), "names", round((out / f"{kind}.txt.gz").stat().st_size / 1e6, 1), "MB")
    chosen = json.loads(THRESHOLDS.read_text())
    thresholds = {label: cut["threshold"] for label, cut in chosen.items() if cut["threshold"] > 0}
    config = {"chunk_chars": CHUNK_CHARS, "max_tokens": EVAL_LENGTH, "thresholds": thresholds}
    (out / "tagger.json").write_text(json.dumps(config, indent=1))


if __name__ == "__main__":
    export(sys.argv[1] if len(sys.argv) > 1 else "release")
