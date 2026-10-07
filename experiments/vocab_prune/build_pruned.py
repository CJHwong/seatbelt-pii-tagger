# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers", "sentencepiece", "protobuf", "numpy", "pyarrow", "huggingface_hub"]
# ///
"""Build the 64,000-piece tokenizer and the two starting checkpoints.

Outputs under build/:
  tokenizer-pruned/   the pruned Unigram tokenizer
  tokenizer-full/     the original XLM-R tokenizer, for the full-vocabulary control
  init-pruned.pt      first 6 MiniLM layers plus the embedding rows for the kept pieces
  init-full.pt        first 6 MiniLM layers plus all 250,037 embedding rows
  build_stats.json    corpus size, forced and chosen counts, coverage before and after

--corpus prepared ranks pieces on data/prepared/train.jsonl, which holds Traditional text,
instead of the Wismut train split. --out picks the output directory, so the experiment's
build/ stays as RESULTS.md describes it. --split-punctuation isolates punctuation as its
own pre-token, so no piece crosses a span boundary at a colon or a bracket.
--extra-corpus adds a JSONL file of texts to the ranking, such as real Traditional text.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast, XLMRobertaTokenizerFast

from data import DATA, eval_texts, minilm, pruning_texts
from prune_lib import pruned_tokenizer_json, select_pieces, with_punctuation_split

TARGET_SIZE = 64_000
KEEP_LAYERS = 6


def piece_frequencies(tokenizer, texts: list[str]) -> Counter:
    counts: Counter = Counter()
    for start in range(0, len(texts), 1000):
        for ids in tokenizer(texts[start : start + 1000], add_special_tokens=False)["input_ids"]:
            counts.update(ids)
    return counts


def truncated_state(state: dict, kept_ids: list[int] | None) -> dict:
    keep = {k: v for k, v in state.items() if not k.startswith("encoder.layer.") or _layer(k) < KEEP_LAYERS}
    if kept_ids is not None:
        keep["embeddings.word_embeddings.weight"] = state["embeddings.word_embeddings.weight"][kept_ids].clone()
    return keep


def _layer(key: str) -> int:
    return int(key.split(".")[2])


def coverage(tokenizer, texts: list[str]) -> dict:
    encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
    tokens = sum(len(ids) for ids in encoded)
    unknown = sum(ids.count(tokenizer.unk_token_id) for ids in encoded)
    characters = sum(len(text) for text in texts)
    return {"tokens": tokens, "unknown": unknown, "tokens_per_char": tokens / characters}


def check_rows(pruned, original, kept_ids: list[int]) -> None:
    """Every new id must name the same piece as the old id whose embedding row it took."""
    new_pieces = pruned.convert_ids_to_tokens(list(range(len(kept_ids))))
    old_pieces = original.convert_ids_to_tokens(kept_ids)
    mismatched = [i for i, (new, old) in enumerate(zip(new_pieces, old_pieces, strict=True)) if new != old]
    if mismatched:
        raise SystemExit(f"{len(mismatched)} pieces point at the wrong embedding row, first at id {mismatched[0]}")


def prepared_texts() -> list[str]:
    with open(DATA / "prepared/train.jsonl", encoding="utf-8") as handle:
        return [json.loads(line)["text"] for line in handle]


def synthetic_heldout_texts() -> list[str]:
    with open(DATA / "synthetic/heldout.jsonl", encoding="utf-8") as handle:
        return [json.loads(line)["text"] for line in handle]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["wismut", "prepared"], default="wismut")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "build")
    parser.add_argument("--split-punctuation", action="store_true", help="isolate punctuation as its own pre-token")
    parser.add_argument("--extra-corpus", type=Path, help="a JSONL file of {text} rows added to the ranking corpus")
    arguments = parser.parse_args()
    build_dir = arguments.out
    build_dir.mkdir(exist_ok=True)
    original = XLMRobertaTokenizerFast.from_pretrained(minilm())
    corpus = prepared_texts() if arguments.corpus == "prepared" else pruning_texts()
    if arguments.extra_corpus:
        with open(arguments.extra_corpus, encoding="utf-8") as handle:
            corpus += [json.loads(line)["text"] for line in handle]
    frequencies = piece_frequencies(original, corpus)
    spec = json.loads(original.backend_tokenizer.to_str())
    kept_ids = select_pieces(spec["model"]["vocab"], frequencies, TARGET_SIZE)

    backend_json = pruned_tokenizer_json(original.backend_tokenizer.to_str(), kept_ids)
    if arguments.split_punctuation:
        backend_json = with_punctuation_split(backend_json)
    (build_dir / "tokenizer-pruned").mkdir(exist_ok=True)
    (build_dir / "tokenizer-pruned" / "tokenizer.json").write_text(backend_json)
    pruned = PreTrainedTokenizerFast(
        tokenizer_file=str(build_dir / "tokenizer-pruned" / "tokenizer.json"),
        **original.special_tokens_map,
    )
    pruned.save_pretrained(build_dir / "tokenizer-pruned")
    original.save_pretrained(build_dir / "tokenizer-full")
    check_rows(pruned, original, kept_ids)

    state = torch.load(Path(minilm()) / "pytorch_model.bin", map_location="cpu")
    torch.save(truncated_state(state, kept_ids), build_dir / "init-pruned.pt")
    torch.save(truncated_state(state, None), build_dir / "init-full.pt")

    samples = eval_texts()
    if (DATA / "synthetic/heldout.jsonl").exists():
        samples["tw_synthetic_heldout"] = synthetic_heldout_texts()
    stats = {
        "corpus": arguments.corpus,
        "split_punctuation": arguments.split_punctuation,
        "extra_corpus": str(arguments.extra_corpus) if arguments.extra_corpus else None,
        "corpus_rows": len(corpus),
        "distinct_pieces_seen": len(frequencies),
        "forced_pieces": sum(1 for piece, _ in spec["model"]["vocab"] if len(piece) == 1) + 5,
        "kept": len(kept_ids),
        "corpus_token_share_kept": sum(frequencies[i] for i in kept_ids) / sum(frequencies.values()),
        "coverage": {
            name: {"full": coverage(original, texts), "pruned": coverage(pruned, texts)}
            for name, texts in samples.items()
        },
    }
    (build_dir / "build_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    json.dump(stats, sys.stdout, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
