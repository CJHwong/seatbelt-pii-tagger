# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch>=2.2,<3", "transformers>=4.40,<6", "tokenizers>=0.15", "sentencepiece", "protobuf",
#     "numpy", "pyyaml", "pyarrow", "coremltools>=8",
# ]
# ///
"""Export one checkpoint to the Core ML package the hook runs on the Apple Neural Engine.

usage: uv run --script export_coreml.py [RUN]   RUN defaults to release; macOS only

Writes build/RUN/model.mlpackage: fp16, one function per input length (length_64,
length_128, length_256, length_512) over one shared copy of the weights. The Neural
Engine computes in fp16 and runs only fixed shapes, so the hook pads each window to the
shortest length that holds it. export_onnx.py writes the other files the hook reads.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import coremltools
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE.parent / p) for p in ("first_run", "vocab_prune")]

from train import RUNS, final_checkpoint, load_checkpoint

LENGTHS = (64, 128, 192, 256, 320, 384, 512)
# The fp32 export masks with finfo(float32).min, which is -inf in fp16 and turns a
# fully masked row into NaN. -1e4 is far enough below any score to zero the softmax.
MASKED = -1e4


class LogitsOnly(torch.nn.Module):
    """The tagger with a plain tensor output and every length-derived input passed in.

    transformers slices its position and segment buffers by the traced length, and
    coremltools folds that length into a constant it cannot cast, so both come in as
    inputs computed from input_ids.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        additive = (1.0 - attention_mask[:, None, None, :].to(torch.float32)) * MASKED
        positions = torch.cumsum(torch.ones_like(input_ids), dim=1) - 1
        return self.model(
            input_ids=input_ids,
            attention_mask=additive,
            position_ids=positions,
            token_type_ids=torch.zeros_like(input_ids),
        ).logits


def convert(model, length: int, out: Path) -> None:
    ids = torch.ones((1, length), dtype=torch.int32)
    traced = torch.jit.trace(LogitsOnly(model), (ids, torch.ones_like(ids)))
    shape = (1, length)
    coremltools.convert(
        traced,
        inputs=[
            coremltools.TensorType(name="input_ids", shape=shape, dtype=np.int32),
            coremltools.TensorType(name="attention_mask", shape=shape, dtype=np.int32),
        ],
        outputs=[coremltools.TensorType(name="logits")],
        convert_to="mlprogram",
        compute_precision=coremltools.precision.FLOAT16,
        minimum_deployment_target=coremltools.target.macOS15,
    ).save(str(out))


def export(run: str) -> Path:
    out = HERE / "build" / run / "model.mlpackage"
    model, _, _ = load_checkpoint(final_checkpoint(RUNS / run))
    model.to("cpu").eval()
    model.config._attn_implementation = "eager"
    descriptor = coremltools.utils.MultiFunctionDescriptor()
    with tempfile.TemporaryDirectory() as scratch:
        for length in LENGTHS:
            single = Path(scratch) / f"length_{length}.mlpackage"
            convert(model, length, single)
            descriptor.add_function(str(single), src_function_name="main", target_function_name=f"length_{length}")
        descriptor.default_function_name = "length_128"
        if out.exists():
            shutil.rmtree(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        coremltools.utils.save_multifunction(descriptor, str(out))
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    export(sys.argv[1] if len(sys.argv) > 1 else "release")
