# /// script
# requires-python = ">=3.12"
# dependencies = ["pyyaml"]
# ///
"""The released model against the external systems, from the saved scores.

usage: uv run --script tables.py   -> runs/tables.md, printed

Columns: the release run raw, the release with the chain agent-seatbelt runs after it, and
the five external systems. Every set sits in one suite. A suite row is the mean of one
metric over that suite's sets, so F1 and recall never share a mean, and clean rows flagged
add up. Each row names its metric and its better direction, and bold is the best in the
row. "ours" marks a set built for this model, "public" a published dataset. The private
transcript sets appear only where their scores exist.
"""

import json
from pathlib import Path

from compare_lib import release_run

HERE = Path(__file__).parent
RUNS = HERE / "runs"
EXTERNAL = {
    "opf": "OpenAI Privacy Filter",
    "redact_model": "Redact model",
    "redact_hook": "Redact hook",
    "rules": "agent-seatbelt rules",
    "gliner": "NVIDIA GLiNER-PII",
}


GROUPS = {
    "Our data": RUNS / "scores.json",
    "Generalization": RUNS / "generalization/scores.json",
    "Blind synthetic": RUNS / "blind/scores.json",
    "Secrets": RUNS / "rules/scores.json",
    "External benchmarks": RUNS / "benchmarks/scores.json",
    "Benchmarks the external systems publish": RUNS / "published/scores.json",
}


# Who made each set: ours (built for this model, or taken from the agent-seatbelt test
# suite) or public (a published dataset).
OURS = {"our_benchmark", "blind_zh-Hant", "blind_zh-Hans", "blind_en", "generated_held_out"}


OURS_PREFIXES = ("perturbed_", "seatbelt_")


def origin(set_name: str) -> str:
    return "ours" if set_name in OURS or set_name.startswith(OURS_PREFIXES) else "public"


# The suite of every set in the summary. A set missing here fails the build of the summary.
SUITES = {
    "Our core": ["our_benchmark", "blind_en", "blind_zh-Hans", "blind_zh-Hant", "perturbed_fullwidth", "perturbed_spaced", "perturbed_ocr", "perturbed_long"],
    "Chinese": ["tw_bench", "zh_bench", "weibo", "wikinews_zhtw"],
    "English and multilingual": ["gretel_en", "tab_echr", "wnut17", "argilla", "spy", "wismut_test_latin"],
    "Secrets": ["creddata", "generated_held_out", "seatbelt_secrets_trained", "seatbelt_secrets_held_out", "seatbelt_clean", "seatbelt_positive",
                "seatbelt_test_cases", "seatbelt_holdout", "seatbelt_final_holdout", "seatbelt_challenge"],
    "Published by the external systems": ["nemotron_test", "pii_masking_300k", "kiji_en", "gretel_finance_en", "wikiann_en", "wikiann_zh",
                                          "multinerd_en", "multinerd_zh", "conll2003_en", "multiconer_en", "multiconer_zh"],
}  # fmt: skip


# Scored on this machine only. The pii-masking-300k licence needs written permission to
# share results, so the published tables leave it out.
LOCAL_ONLY = {"pii_masking_300k"}

SKIP = {
    "perturbed_fullwidth_original",
    "perturbed_spaced_original",
    "perturbed_ocr_original",
    "perturbed_long_original",
}


def metric_of(report: dict) -> tuple[str, float, bool] | None:
    if "typed_overlap" in report:
        return "typed overlap F1 (higher is better)", report["typed_overlap"]["f1"], True
    if "any_overlap" in report:
        return "any overlap F1 (higher is better)", report["any_overlap"]["f1"], True
    if "person_recall" in report:
        return "person recall (higher is better)", report["person_recall"], True
    if "flags_per_10k_chars" in report:
        return "flags per 10k chars (lower is better)", report["flags_per_10k_chars"], False
    if "row_recall" in report:
        return "row recall (higher is better)", report["row_recall"], True
    return None


def cells(values: list[float | None], higher: bool, digits: int) -> list[str]:
    present = [v for v in values if v is not None]
    best = (max if higher else min)(present) if present else None
    out = []
    for value in values:
        if value is None:
            out.append("")
            continue
        text = f"{value:.{digits}f}" if digits else f"{value:g}"
        out.append(f"**{text}**" if value == best else text)
    return out


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(r) + " |" for r in rows),
        "",
    ]


def rows_of(path: Path, names: list[str]) -> list[dict]:
    """One record per table row: its set, metric, better direction, and one value per system."""
    scores = json.loads(path.read_text())
    present = [n for n in names if n in scores]
    found_rows = []
    for set_name, report in scores[present[0]].items():
        if set_name in SKIP:
            continue
        no_gold = bool(report.get("clean_rows")) and not report.get("gold_by_type")
        found = None if no_gold else metric_of(report)  # a clean set has no F1
        base = {"set": set_name, "origin": origin(set_name)}
        if found:
            label, _, higher = found
            values = {
                n: metric_of(scores[n][set_name])[1]
                for n in present
                if set_name in scores[n] and metric_of(scores[n][set_name])
            }
            found_rows.append(
                {
                    **base,
                    "table": "main",
                    "label": label,
                    "higher": higher,
                    "digits": 3 if "flags" not in label else 1,
                    "values": values,
                }
            )
        if "typed_overlap" in report and found:
            values = {n: scores[n][set_name]["typed_overlap"]["precision"] for n in present if set_name in scores[n]}
            found_rows.append(
                {
                    **base,
                    "table": "precision",
                    "label": "typed overlap precision (higher is better)",
                    "higher": True,
                    "digits": 3,
                    "values": values,
                }
            )
        if report.get("clean_rows"):
            values = {n: scores[n][set_name]["clean_flagged"] for n in present if set_name in scores[n]}
            found_rows.append(
                {
                    **base,
                    "table": "clean",
                    "label": "clean rows flagged (lower is better)",
                    "higher": False,
                    "digits": 0,
                    "values": values,
                    "of": report["clean_rows"],
                }
            )
    return found_rows


def row_cells(row: dict, names: list[str]) -> list[str]:
    return cells([row["values"].get(n) for n in names], row["higher"], row["digits"])


def short_metric(label: str) -> str:
    return (
        label.replace(" (higher is better)", " ↑")
        .replace(" (lower is better)", " ↓")
        .replace("typed overlap ", "typed ")
    )


def suite_rows(found_rows: list[dict], names: list[str]) -> list[list[str]]:
    """Per suite and metric, the mean over the sets that report it, and how many sets that is."""
    lines = []
    for suite, members in SUITES.items():
        mine = [r for r in found_rows if r["set"] in members and r["table"] != "precision"]
        for label in dict.fromkeys(r["label"] for r in mine):
            same = [r for r in mine if r["label"] == label]
            if label.startswith("clean rows"):  # a total, not a mean: flagged of all clean rows
                values = {n: sum(r["values"][n] for r in same) for n in names if all(n in r["values"] for r in same)}
                row = {"values": values, "higher": False, "digits": 0}
                lines.append(
                    [
                        suite,
                        f"clean rows flagged, of {sum(r['of'] for r in same)} ↓",
                        str(len(same)),
                        *row_cells(row, names),
                    ]
                )
                continue
            values = {
                n: sum(r["values"][n] for r in same) / len(same) for n in names if all(n in r["values"] for r in same)
            }
            row = {"values": values, "higher": same[0]["higher"], "digits": same[0]["digits"]}
            lines.append([suite, f"mean {short_metric(label)}", str(len(same)), *row_cells(row, names)])
    return lines


def transcript_rows(names: list[str]) -> list[list[str]]:
    lines = []
    for name, path in (("dev", "report.json"), ("test", "test/report.json")):
        if not (RUNS / "transcripts" / path).exists():  # private sets; most clones lack them
            continue
        report = json.loads((RUNS / "transcripts" / path).read_text())
        for metric in ("f1", "recall", "precision"):
            row = {
                "values": {n: report[n]["judged"][metric] for n in names if n in report},
                "higher": True,
                "digits": 3,
            }
            lines.append([f"transcript_{name}", f"{metric} ↑", *row_cells(row, names)])
        row = {
            "values": {n: report[n]["flags_per_10k_chars"] for n in names if n in report},
            "higher": False,
            "digits": 1,
        }
        lines.append([f"transcript_{name}", "flags per 10k chars ↓", *row_cells(row, names)])
    return lines


def columns() -> dict[str, str]:
    run = release_run()
    return {run: "raw", f"{run}_post_pub_plc_off_thr_keys": "+ chain", **EXTERNAL}


def report(groups: dict[str, list[dict]]) -> list[str]:
    names = list(columns())
    found_rows = [r for rows in groups.values() for r in rows]
    unplaced = {r["set"] for r in found_rows} - {s for members in SUITES.values() for s in members}
    if unplaced:
        raise SystemExit(f"<tables_failed>sets in no suite: {sorted(unplaced)}</tables_failed>")
    heads = list(columns().values())
    lines = [
        "# seatbelt-pii-tagger against the external systems",
        "",
        "↑ means higher is better, ↓ lower is better.",
        "",
    ]
    lines += ["## Suites", ""] + table(["suite", "metric", "sets", *heads], suite_rows(found_rows, names))
    for suite, members in SUITES.items():
        rows = [r for member in members for r in found_rows if r["set"] == member and r["table"] != "precision"]
        body = [
            [
                r["set"] + (f" (of {r['of']})" if r["table"] == "clean" else ""),
                r["origin"],
                short_metric(r["label"]),
                *row_cells(r, names),
            ]
            for r in rows
        ]
        lines += [f"## {suite}", ""] + table(["set", "by", "metric", *heads], body)
    transcripts = transcript_rows(names)
    if transcripts:
        lines += ["## Transcripts (private)", ""] + table(["set", "metric", *heads], transcripts)
    return lines


def main() -> None:
    names = list(columns())
    groups = {
        title: [r for r in rows_of(path, names) if r["set"] not in LOCAL_ONLY]
        for title, path in GROUPS.items()
        if path.exists()
    }
    lines = report(groups)
    (RUNS / "tables.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
