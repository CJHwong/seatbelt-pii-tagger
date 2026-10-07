# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers", "sentencepiece", "protobuf", "numpy", "pyarrow", "pyyaml"]
# ///
"""Turn the checked coding-agent session rows into a training split.

usage: uv run --script prepare.py   -> train.jsonl, prepare_stats.json

The gates are the developer-work source's (data/synthetic/llm_devwork/prepare.py). Two
differ: rows with no mark are kept up to NEGATIVE_SHARE, because the product and status
words in them are a third of the point here, and the evaluation-overlap gate also covers
the transcript dev and test windows, because these rows imitate agent sessions.
"""

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).parent
DEVWORK = HERE.parent / "llm_devwork" / "prepare.py"
TRANSCRIPTS = HERE.parents[2] / "experiments/compare/runs/transcripts"
NEGATIVE_SHARE = 0.25


def transcript_texts(base: Path = TRANSCRIPTS) -> list[str]:
    """The dev and test windows, or nothing on a machine without them."""
    paths = [base / "sample.jsonl", base / "test" / "sample.jsonl"]
    return [json.loads(line)["text"] for path in paths if path.exists() for line in path.open(encoding="utf-8")]


def main() -> None:
    spec = importlib.util.spec_from_file_location("devwork_prepare", DEVWORK)
    devwork = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(devwork)
    devwork.build(HERE, NEGATIVE_SHARE, transcript_texts())


if __name__ == "__main__":
    main()
