"""The comparison sets, with gold in our types and the spans each set scores as don't-care."""

import json
import random
import sys
from functools import cache
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(EXPERIMENTS / "first_run"), str(EXPERIMENTS / "vocab_prune"), str(EXPERIMENTS.parent / "scripts")]

from compare_lib import mark_leaked
from data import zh_bench_rows
from train import BENCHMARK, SEED, read_jsonl, tw_bench, wismut_test

ZH_BENCH_ROWS = 1_000


BLIND = EXPERIMENTS.parent / "blind"


@cache
def generalization_sets() -> dict[str, list[dict]]:
    """Sets that share no source with training: external data, and the perturbed copies
    of our benchmark, each variant beside the same rows unperturbed."""
    from external import external_sets

    sets = external_sets()
    for row in read_jsonl(BLIND / "perturbed.jsonl"):
        sets.setdefault(f"perturbed_{row['variant']}", []).append({**row, "dont_care": row["ignore"], "kind": "typed"})
    return sets


@cache
def blind_sets() -> dict[str, list[dict]]:
    """The rows another model family wrote and LLM reviewers checked, one set per language."""
    blind = read_jsonl(BLIND / "blind.jsonl")
    return {
        f"blind_{language}": [
            {**row, "dont_care": row["ignore"], "kind": "typed"} for row in blind if row["language"] == language
        ]
        for language in ("zh-Hant", "zh-Hans", "en")
    }


def rules_group() -> dict[str, list[dict]]:
    from seatbelt import rules_sets

    return rules_sets()


def benchmarks_group() -> dict[str, list[dict]]:
    from benchmarks import benchmark_sets

    return benchmark_sets()


def published_group() -> dict[str, list[dict]]:
    from published import published_sets

    return published_sets()


LEAKED = Path(__file__).parent / "runs/leaked.json"  # written by leak_check.py


def marked(load):
    """The group, with `leaked` on every row that leak_check.py found in the training build."""
    leaked = json.loads(LEAKED.read_text()) if LEAKED.exists() else {}
    return lambda: mark_leaked(load(), leaked)


GROUPS = {
    "comparison": marked(lambda: comparison_sets()),
    "generalization": marked(generalization_sets),
    "blind": marked(blind_sets),
    "rules": marked(rules_group),
    "benchmarks": marked(benchmarks_group),
    "published": marked(published_group),
}


@cache
def comparison_sets() -> dict[str, list[dict]]:
    zh = zh_bench_rows()
    return {
        "our_benchmark": [{**row, "dont_care": row["ignore"]} for row in read_jsonl(BENCHMARK)],
        "tw_bench": tw_bench(),
        "zh_bench": random.Random(SEED).sample(zh, min(len(zh), ZH_BENCH_ROWS)),
        "wismut_test_latin": wismut_test()["latin"],
    }
