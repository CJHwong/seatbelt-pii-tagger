# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch>=2.2,<3", "transformers>=4.40,<6", "tokenizers>=0.15", "sentencepiece", "protobuf",
#     "numpy", "pyyaml", "pyarrow", "onnxruntime>=1.17", "huggingface_hub", "ai-edge-litert>=2.0",
# ]
# ///
"""Check the agent-seatbelt tagger backend gives the spans this repository scored.

usage: uv run --script parity.py HOOKS_PII_DIR EXPORT_DIR [SYSTEM]

Runs the hook's tagger and its chain (pii_tagger.TaggerModel.tag, then apply_chain) on
every row of every comparison group, and compares the spans, by start, end and label,
with runs/[GROUP/]SYSTEM.jsonl. SYSTEM defaults to the int8 export with the chain,
release_onnx8_post_pub_plc_off_thr_keys. Prints the rows that differ per group; the
texts and values stay out of the output.
"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
COMPARE = HERE.parent / "compare"
sys.path[:0] = [str(COMPARE), str(HERE.parent / "first_run"), str(HERE.parent / "vocab_prune")]
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from predict import run_dir
from sets import GROUPS


def key(spans: list[dict]) -> list[tuple[int, int, str]]:
    return sorted((span["start"], span["end"], span["label"]) for span in spans)


def main(hooks: str, export: str, system: str) -> None:
    sys.path.insert(0, hooks)
    os.environ["PII_TAGGER_DIR"] = export
    from pii_tagger import TaggerModel, asset_dir
    from pii_tagger_rules import apply_chain

    model = TaggerModel(asset_dir())
    total = differing = 0
    for group, load in GROUPS.items():
        expected = [json.loads(line) for line in (run_dir(group) / f"{system}.jsonl").open(encoding="utf-8")]
        rows = [(name, index, row) for name, set_rows in load().items() for index, row in enumerate(set_rows)]
        if len(rows) != len(expected):
            raise SystemExit(f"{group}: {len(rows)} rows against {len(expected)} scored")
        found = 0
        for (name, index, row), scored in zip(rows, expected, strict=True):
            if (name, index) != (scored["set"], scored["index"]):
                raise SystemExit(f"{group}: row order differs at {name} {index}")
            spans = apply_chain(model.tag(row["text"]), row["text"], model.persons, model.places, model.thresholds)
            found += key(spans) != key(scored["spans"])
        print(f"{group}: {found} of {len(rows)} rows differ", flush=True)
        total, differing = total + len(rows), differing + found
    print(f"all groups: {differing} of {total} rows differ")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "release_onnx8_post_pub_plc_off_thr_keys")
