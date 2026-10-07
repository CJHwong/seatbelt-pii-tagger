#!/bin/zsh
# usage: chain.sh SYSTEM   -> runs/[GROUP/]SYSTEM_post_pub_plc_off_thr_keys.jsonl for every group
#
# The chain for one run or ensemble predict.py knows: predictions for every group, then
# the post rules, the public persons list at floor 20, the places list at floor 20, the
# office-title rule, the per-label thresholds in thresholds.json, and handles in username
# fields. Score the result with score.py, whose SYSTEMS must name the chain's outputs.
set -e
cd "$(dirname "$0")"
SYSTEM=$1
export HF_HUB_OFFLINE=1
for GROUP in comparison generalization blind rules benchmarks published; do
  uv run --script predict.py "$SYSTEM" "$GROUP"
done
uv run --script post_rules.py apply "$SYSTEM"
uv run --script public_filter.py apply persons "${SYSTEM}_post" 20
uv run --script public_filter.py apply places "${SYSTEM}_post_pub" 20
uv run --script public_filter.py apply offices "${SYSTEM}_post_pub_plc" 0
uv run --script label_thresholds.py apply "${SYSTEM}_post_pub_plc_off" thresholds.json
uv run --script post_rules.py keys "${SYSTEM}_post_pub_plc_off_thr"
echo "chain done: ${SYSTEM}_post_pub_plc_off_thr_keys"
