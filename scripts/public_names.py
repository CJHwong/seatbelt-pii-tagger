# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Fetch the names of notable people and places from Wikidata (CC0) for the public filter.

usage: uv run scripts/public_names.py
Writes data/gazetteer/public_persons.tsv and public_places.tsv with one row per name: the
Wikidata item, its sitelink count, the language code, and whether the name is a label or
an alias. People are humans (P31 Q5) and places are items with coordinates (P625), each
with more than its FLOORS sitelinks; the filter picks a higher floor.

The query runs on QLever (qlever.dev), which scans all 13M humans quickly. QLever
compares wikibase:sitelinks as text unless the value is cast, so the query casts it.
"""

import csv
import io
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/gazetteer"
ENDPOINT = "https://qlever.dev/api/wikidata"
KINDS = {"persons": "?item wdt:P31 wd:Q5 .", "places": "?item wdt:P625 ?coordinates ."}
FLOORS = {"persons": 10, "places": 20}
LANGUAGES = ("en", "zh", "zh-hans", "zh-hant", "zh-tw", "zh-cn", "zh-hk")
PREDICATES = {"label": "rdfs:label", "alias": "skos:altLabel"}
PREFIXES = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""


def query(kind: str, predicate: str) -> str:
    languages = " || ".join(f'LANG(?name) = "{code}"' for code in LANGUAGES)
    return f"""{PREFIXES}SELECT DISTINCT ?item ?n ?name WHERE {{
  {KINDS[kind]}
  ?item wikibase:sitelinks ?n .
  FILTER(xsd:integer(?n) > {FLOORS[kind]})
  ?item {predicate} ?name .
  FILTER({languages})
}}"""


def fetch(sparql: str) -> list[dict]:
    request = urllib.request.Request(
        f"{ENDPOINT}?{urllib.parse.urlencode({'query': sparql})}",
        headers={"Accept": "text/tab-separated-values", "User-Agent": "seatbelt-pii-tagger/2026.10"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        text = response.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE))


def parsed(row: dict, kind: str) -> tuple[str, int, str, str, str]:
    """TSV cells hold RDF terms: <.../Q76>, 340, "Barack Obama"@en."""
    literal = row["?name"]
    value, _, language = literal.rpartition("@")
    name = value[1:-1].replace('\\"', '"')
    return row["?item"].rsplit("/", 1)[1].rstrip(">"), int(row["?n"]), language, kind, name


def write_kind(kind: str) -> None:
    rows = []
    for name_kind, predicate in PREDICATES.items():
        fetched = fetch(query(kind, predicate))
        if not fetched:
            raise SystemExit(f"the {kind} {name_kind} query returned no rows")
        rows += [parsed(row, name_kind) for row in fetched]
        print(f"{kind} {name_kind}: {len(fetched)} names", file=sys.stderr)
    out = OUT / f"public_{kind}.tsv"
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["item", "sitelinks", "language", "kind", "name"])
        writer.writerows(rows)
    print(f"{len(rows)} rows -> {out}", file=sys.stderr)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for kind in sys.argv[1:] or KINDS:
        write_kind(kind)


if __name__ == "__main__":
    main()
