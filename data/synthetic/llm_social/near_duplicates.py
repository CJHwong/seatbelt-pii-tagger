"""Find rows that share a frame: high shingle overlap once marked values and digits are blanked.

usage: python3 near_duplicates.py THRESHOLD FILE...   -> per writer: rows in a near-duplicate cluster

Shingles are 3-word sequences for Latin rows and 4-character sequences for Chinese rows.
Two rows are near duplicates when their Jaccard overlap is at least THRESHOLD. skeletons.py
catches exact frames; this catches frames whose nouns, cities or items change.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

MARK = re.compile(r"\[\[\w+:.+?\]\]", re.DOTALL)
DIGITS = re.compile(r"[0-9０-９]+")
WORD = re.compile(r"\w+")


def shingles(row: dict) -> set:
    text = DIGITS.sub("0", MARK.sub("@", row["text"])).lower()
    if row["script"] == "latin":
        words = WORD.findall(text)
        return {tuple(words[i : i + 3]) for i in range(len(words) - 2)}
    flat = re.sub(r"\s+", "", text)
    return {flat[i : i + 4] for i in range(len(flat) - 3)}


def clusters(rows: list[dict], threshold: float) -> list[list[int]]:
    sets = [shingles(row) for row in rows]
    parent = list(range(len(rows)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            union = len(sets[i] | sets[j])
            if union and len(sets[i] & sets[j]) / union >= threshold:
                parent[root(i)] = root(j)
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[root(i)].append(i)
    return [group for group in groups.values() if len(group) > 1]


def main() -> None:
    threshold = float(sys.argv[1])
    by_writer = defaultdict(list)
    for path in sys.argv[2:]:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                by_writer[row["id"].split("-")[0]].append(row)
    for writer, rows in sorted(by_writer.items()):
        found = clusters(rows, threshold)
        extra = sum(len(group) - 1 for group in found)
        print(f"{writer}: {len(rows)} rows, {len(found)} clusters, {extra} rows beyond the first of a cluster")
        for group in sorted(found, key=len, reverse=True)[:2]:
            print(f"   x{len(group)}: {rows[group[0]]['text'][:70]!r}")


if __name__ == "__main__":
    main()
