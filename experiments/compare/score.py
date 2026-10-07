# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers", "numpy", "pyarrow", "pyyaml"]
# ///
"""Score every runs/SYSTEM.jsonl on the same rules and write runs/report.md.

Rules applied to every system alike:
  - a prediction on a set's don't-care span counts neither way;
  - on tw-PII-bench, repeats of a gold name are set aside, as in our own evaluation;
  - dates count on neither side (compare_lib.row_counts says why);
  - zh-bench labels are its own, so it gets the untyped score only.

usage: uv run --script score.py [GROUP]   GROUP is comparison (default), generalization or blind.
In the generalization group a row's `kind` decides its scores (external.py says why):
typed rows get the scores above, recall rows get person recall, clean rows get flags
per 10,000 characters, and every flag on a clean row is written out for review.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from compare_lib import local_systems, normalized, outside, overlaps, release_run, row_counts, summary, unleaked
from seatbelt import row_label_hit
from sets import GROUPS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "first_run"))
from run_lib import drop_name_repeats

HERE = Path(__file__).parent
# The released model, raw and after the chain, its int8 export, and the systems it is compared
# with. A system scores when its runs/[GROUP/]SYSTEM.jsonl exists. local_systems() may name
# another list, in its own order.
RELEASE = release_run()
PUBLIC_SYSTEMS = [
    RELEASE,
    f"{RELEASE}_post_pub_plc_off_thr_keys",
    f"{RELEASE}_onnx8",
    f"{RELEASE}_onnx8_post_pub_plc_off_thr_keys",
    "opf",
    "redact_model",
    "redact_hook",
    "rules",
    "gliner",
]
SYSTEMS = local_systems().get("score_systems") or PUBLIC_SYSTEMS
TYPED_SETS = {"our_benchmark", "tw_bench", "wismut_test_latin"}


def run_dir(group: str) -> Path:
    return HERE / "runs" if group == "comparison" else HERE / "runs" / group


def load_predictions(system: str, group: str) -> dict[str, dict[int, list[dict]]]:
    by_set: dict[str, dict[int, list[dict]]] = defaultdict(dict)
    with open(run_dir(group) / f"{system}.jsonl", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            spans = [{**s, "label": normalized(s["label"])} for s in record["spans"]]
            by_set[record["set"]][record["index"]] = spans
    return by_set


def cleaned(name: str, row: dict, spans: list[dict]) -> list[dict]:
    spans = outside(spans, row.get("dont_care", []))
    if name == "tw_bench":
        spans, _ = drop_name_repeats(spans, row["text"], row["entities"])
    return spans


def score_set(name: str, rows: list[dict], predictions: dict[int, list[dict]]) -> dict:
    counts: Counter = Counter()
    recall_by_type: dict[str, Counter] = defaultdict(Counter)
    clean_rows = clean_flagged = 0
    for index, row in enumerate(rows):
        spans = cleaned(name, row, predictions.get(index, []))
        counts += row_counts(row["entities"], spans)
        for gold in row["entities"]:
            recall_by_type[gold["label"]]["gold"] += 1
            recall_by_type[gold["label"]]["hit"] += any(overlaps(gold, p) for p in spans)
        if not row["entities"]:
            clean_rows += 1
            clean_flagged += any(p["label"] != "date" for p in spans)
    report = summary(counts)
    kind = rows[0].get("kind") if rows else None
    if kind == "recall":
        return {**report_recall(rows, predictions, name), "recall_by_type": {}, "gold_by_type": {}}
    if kind == "row_label":
        return report_row_label(rows, predictions, name)
    if kind == "clean":
        return report_clean(rows, predictions, name)
    if name not in TYPED_SETS and kind != "typed":
        report = {"any_overlap": report["any_overlap"]}
    report["recall_by_type"] = {t: c["hit"] / c["gold"] for t, c in recall_by_type.items()}
    report["gold_by_type"] = {t: c["gold"] for t, c in recall_by_type.items()}
    report["clean_rows"], report["clean_flagged"] = clean_rows, clean_flagged
    return report


def report_recall(rows: list[dict], predictions: dict[int, list[dict]], name: str) -> dict:
    """Person recall only: these sets mark persons alone, public figures included."""
    gold = typed_hit = any_hit = 0
    for index, row in enumerate(rows):
        spans = cleaned(name, row, predictions.get(index, []))
        for person in row["entities"]:
            gold += 1
            typed_hit += any(overlaps(person, p) and p["label"] == "person" for p in spans)
            any_hit += any(overlaps(person, p) for p in spans)
    return {"persons": gold, "person_recall": typed_hit / gold, "any_label_recall": any_hit / gold}


def report_row_label(rows: list[dict], predictions: dict[int, list[dict]], name: str) -> dict:
    """Rows labelled per row, not per span: a hit is any span with an expected label."""
    hits = sum(
        row_label_hit(row["expected"], cleaned(name, row, predictions.get(index, []))) for index, row in enumerate(rows)
    )
    return {"rows": len(rows), "row_recall": hits / len(rows) if rows else 0.0}


def report_clean(rows: list[dict], predictions: dict[int, list[dict]], name: str) -> dict:
    """No gold: every flag is a candidate false alarm. Dates are left out, as everywhere."""
    flags = [
        {"row": index, "label": p["label"], "value": row["text"][p["start"] : p["end"]], **p}
        for index, row in enumerate(rows)
        for p in cleaned(name, row, predictions.get(index, []))
        if p["label"] != "date"
    ]
    characters = sum(len(row["text"]) for row in rows)
    return {
        "rows": len(rows),
        "rows_flagged": len({flag["row"] for flag in flags}),
        "flags_per_10k_chars": 10_000 * len(flags) / characters,
        "flags_by_label": dict(Counter(flag["label"] for flag in flags)),
        "flags": flags,
    }


def generalization_markdown(results: dict[str, dict[str, dict]], sets: dict[str, list[dict]]) -> str:
    systems = list(results)
    header = lambda first: ["| " + first + " | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
    kinds = {name: rows[0].get("kind") for name, rows in sets.items() if rows}
    typed = [n for n, k in kinds.items() if k == "typed" and not n.startswith("perturbed_")]
    lines = [
        "Typed sets, F1 unless the row says otherwise. F1 and recall: higher is better. Clean rows flagged: lower is better:",
        "",
    ] + header("set, metric")
    for name in typed:
        for metric in ("typed_strict", "typed_overlap", "any_overlap"):
            lines.append(
                f"| {name}, {metric} | " + " | ".join(f"{results[s][name][metric]['f1']:.3f}" for s in systems) + " |"
            )
        if results[systems[0]][name]["clean_rows"]:
            cells = [f"{results[s][name]['clean_flagged']} of {results[s][name]['clean_rows']}" for s in systems]
            lines.append(f"| {name}, clean rows flagged (lower is better) | " + " | ".join(cells) + " |")
    perturbed = []
    for name in typed_perturbed(kinds):
        cells = []
        for s in systems:
            after = results[s][f"perturbed_{name}"]["typed_overlap"]["f1"]
            before = results[s][f"perturbed_{name}_original"]["typed_overlap"]["f1"]
            cells.append(f"{after:.3f} ({after - before:+.3f})")
        perturbed.append(f"| {name}, {len(sets[f'perturbed_{name}'])} rows | " + " | ".join(cells) + " |")
    recall = []
    for name in [n for n, k in kinds.items() if k == "recall"]:
        cells = [
            f"{results[s][name]['person_recall']:.2f} (any label {results[s][name]['any_label_recall']:.2f})"
            for s in systems
        ]
        recall.append(f"| {name}, {results[systems[0]][name]['persons']} persons | " + " | ".join(cells) + " |")
    clean = []
    for name in [n for n, k in kinds.items() if k == "clean"]:
        cells = [
            f"{results[s][name]['flags_per_10k_chars']:.1f} per 10k chars, {results[s][name]['rows_flagged']} of {results[s][name]['rows']} rows"
            for s in systems
        ]
        clean.append(f"| {name} | " + " | ".join(cells) + " |")
    row_label = [
        f"| {name}, {results[systems[0]][name]['rows']} rows | "
        + " | ".join(f"{results[s][name]['row_recall']:.2f}" for s in systems)
        + " |"
        for name, kind in kinds.items()
        if kind == "row_label"
    ]
    sections = [
        (
            "Rows labelled per row, share of rows with a span of the expected label (higher is better):",
            "set",
            row_label,
        ),
        (
            "Perturbed benchmark, typed overlap F1 (higher is better), perturbed against the same rows unperturbed:",
            "variant",
            perturbed,
        ),
        ("Person recall on real text (higher is better); gold includes public figures, so recall only:", "set", recall),
        ("Clean text with no gold, flags that are not dates (lower is better):", "set", clean),
    ]
    for title, first, body in sections:
        if body:
            lines += ["", title, "", *header(first), *body]
    return "\n".join(lines) + "\n"


def typed_perturbed(kinds: dict[str, str]) -> list[str]:
    return [n.removeprefix("perturbed_") for n in kinds if n.startswith("perturbed_") and not n.endswith("_original")]


def markdown(results: dict[str, dict[str, dict]]) -> str:
    systems = list(results)
    lines = ["| set | metric | " + " | ".join(systems) + " |", "|---|---|" + "---|" * len(systems)]
    for name in GROUPS["comparison"]():
        for metric in ("typed_strict", "typed_overlap", "any_overlap"):
            if metric not in results[systems[0]][name]:
                continue
            cells = [f"{results[s][name][metric]['f1']:.3f}" for s in systems]
            lines.append(f"| {name} | {metric} F1 (higher is better) | " + " | ".join(cells) + " |")
        if results[systems[0]][name]["clean_rows"]:
            cells = [f"{results[s][name]['clean_flagged']} of {results[s][name]['clean_rows']}" for s in systems]
            lines.append(f"| {name} | clean rows flagged (lower is better) | " + " | ".join(cells) + " |")
    for name in ("our_benchmark", "tw_bench"):
        gold = results[systems[0]][name]["gold_by_type"]
        lines += ["", f"Recall by gold type on {name}, any label, overlap:", ""]
        lines += ["| type | gold | " + " | ".join(systems) + " |", "|---|---|" + "---|" * len(systems)]
        for label in sorted(gold, key=lambda t: -gold[t]):
            cells = [f"{results[s][name]['recall_by_type'][label]:.2f}" for s in systems]
            lines.append(f"| {label} | {gold[label]} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    group = sys.argv[1] if len(sys.argv) > 1 else "comparison"
    sets, out = GROUPS[group](), run_dir(group)
    present = [s for s in SYSTEMS if (out / f"{s}.jsonl").exists()]
    results = {}
    for system in present:
        predictions = load_predictions(system, group)
        results[system] = {name: score_set(name, *unleaked(rows, predictions[name])) for name, rows in sets.items()}
        for name, report in results[system].items():
            if "flags" in report:
                flags = report.pop("flags")
                (out / f"{system}.{name}.flags.jsonl").write_text(
                    "".join(json.dumps(f, ensure_ascii=False) + "\n" for f in flags)
                )
    (out / "scores.json").write_text(json.dumps(results, indent=1))
    report = markdown(results) if group == "comparison" else generalization_markdown(results, sets)
    (out / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
