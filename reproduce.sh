#!/usr/bin/env bash
# Rebuild the release from public data: fetch, build, train, run the chain, score, export.
#
# usage: ./reproduce.sh
#   TRAIN_STEPS   training steps, 10000 for the release (a smaller number makes a quick check)
#   RELEASE_DATA  a local copy of the release data folder; without it the folder is
#                 downloaded from the Hugging Face dataset CJHwong/seatbelt-pii-tagger-data
#
# Needs uv and git. Training runs on Apple MPS when present, else on the CPU.
# The private transcript sources are skipped, so the result differs from the release:
# see the README.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$PWD
STEPS=${TRAIN_STEPS:-10000}
DATA_REPO=CJHwong/seatbelt-pii-tagger-data
DATA_REVISION=2026.10

echo "== 1. public datasets and external sources"
uv run scripts/fetch_data.py

echo "== 2. release data: the synthetic sets and the public names lists"
if [ -z "${RELEASE_DATA:-}" ]; then
  RELEASE_DATA=$ROOT/data/release_download
  uvx --from huggingface_hub hf download "$DATA_REPO" --repo-type dataset --revision "$DATA_REVISION" \
    --local-dir "$RELEASE_DATA"
fi
mkdir -p data/synthetic data/gazetteer
cp -Rn "$RELEASE_DATA"/synthetic/. data/synthetic/  # -n keeps a file that is already there
cp -n "$RELEASE_DATA"/gazetteer/*.tsv data/gazetteer/

echo "== 3. evaluation inputs"
(cd experiments/compare && uv run --with faker --with pyarrow python benchmarks.py build-spy)
uv run scripts/traditional_text.py

echo "== 4. training set and vocabulary"
uv run scripts/build_training_set.py
(cd experiments/vocab_prune && uv run --script build_pruned.py --corpus prepared --split-punctuation \
  --extra-corpus ../../data/wikimedia/wiki_traditional/texts.jsonl --out build-split-tw)

echo "== 5. train $STEPS steps"
(cd experiments/first_run && uv run --script train.py train release "$STEPS" \
  --lr 4.2e-4 --batch 64 --bf16 --build build-split-tw --seed 7 --probe-steps 1000 \
  --cap synthetic/rule-secrets:zh-Hans:1250 \
  --source-share synthetic/llm-social:0.1 --source-share synthetic/dev-transcripts:0.05 \
  --source-share synthetic/llm-devwork:0.03 --source-share synthetic/llm-agentwork:0.03)

echo "== 6. chain, scores and tables"
bash experiments/compare/chain.sh release
(cd experiments/compare && for group in comparison generalization blind rules benchmarks published; do
  HF_HUB_OFFLINE=1 uv run --script score.py "$group" > "runs/score.$group.log"
done && uv run --script tables.py)

echo "== 7. export"
(cd experiments/export && uv run --script export_onnx.py release)
echo "done: experiments/export/build/release, experiments/compare/runs/tables.md"
