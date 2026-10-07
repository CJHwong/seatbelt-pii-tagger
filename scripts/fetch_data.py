#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=0.34", "pyyaml>=6"]
# ///
"""Download every dataset and external source in datasets.yaml into a local data root.

Idempotent: a dataset already present at the recorded revision is skipped. Run with
--force to re-download. Writes data/INVENTORY.json recording, for each dataset, the
pinned revision and the real on-disk size.

An `external:` entry is either a list of files, each checked against its sha256, or a
git checkout at a pinned commit. A file whose sha256 matches is skipped. A checkout at
another commit is reported and left as it is. An `optional: true` entry is fetched only
when --only names it. --role filters the `datasets:` list only. A dataset whose repo has an
`external:` entry is fetched by that entry, not as a full snapshot.

Usage:
    uv run scripts/fetch_data.py                    # everything
    uv run scripts/fetch_data.py --role eval        # eval sets only
    uv run scripts/fetch_data.py --only nvidia/Nemotron-PII
    uv run scripts/fetch_data.py --only pii-masking-300k   # an optional external source
    uv run scripts/fetch_data.py --force            # ignore what is already there
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Must be set before huggingface_hub is imported, or the progress bars still attach
# themselves to every download and bury the summary this script exists to print.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import yaml
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "datasets.yaml"
INVENTORY = ROOT / "data" / "INVENTORY.json"


def load_manifest() -> tuple[Path, list[dict], list[dict]]:
    with MANIFEST.open() as handle:
        manifest = yaml.safe_load(handle)
    root = ROOT / (os.environ.get("PII_MODEL_DATA_ROOT") or manifest["data_root_default"])  # an absolute value wins
    return root, manifest["datasets"], manifest.get("external", [])


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def entry_for(repo_id: str) -> dict:
    if not INVENTORY.exists():
        return {}
    return json.loads(INVENTORY.read_text()).get(repo_id, {})


def write_inventory(records: dict) -> None:
    INVENTORY.parent.mkdir(parents=True, exist_ok=True)
    INVENTORY.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")


def fetch_one(api: HfApi, spec: dict, root: Path, force: bool, records: dict) -> str:
    repo_id = spec["id"]
    target = root / repo_id
    previous = entry_for(repo_id)

    try:
        info = api.dataset_info(repo_id, revision=spec.get("revision"), files_metadata=False)
        revision = info.sha  # the pinned commit when datasets.yaml names one, else the latest
    except GatedRepoError:
        return "GATED (needs manual approval on Hugging Face, then re-run this script)"
    except RepositoryNotFoundError:
        return "MISSING (repository not found, the id may have changed)"
    except HfHubHTTPError as error:
        return f"HTTP {error.response.status_code if error.response else '?'}"

    if not force and previous.get("revision") == revision and target.exists():
        size = previous.get("bytes", 0)
        records[repo_id] = previous
        return f"SKIP (already at {revision[:8]}, {size / 1e6:.1f} MB)"

    started = time.time()
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=str(target),
            max_workers=8,
        )
    except GatedRepoError:
        return "GATED (needs manual approval on Hugging Face, then re-run this script)"
    except HfHubHTTPError as error:
        return f"FAILED (HTTP {error.response.status_code if error.response else '?'})"

    size = dir_size(target)
    records[repo_id] = {
        "revision": revision,
        "bytes": size,
        "files": sum(1 for f in target.rglob("*") if f.is_file()),
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "licence": spec.get("licence"),
        "role": spec.get("role"),
    }
    return f"OK ({size / 1e6:.1f} MB, {time.time() - started:.0f}s, revision {revision[:8]})"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(spec: dict, file: dict, dest: Path) -> None:
    """One file at the pinned revision, from the Hugging Face dataset or the raw GitHub URL."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if spec["kind"] == "hf":
        cached = hf_hub_download(spec["repo"], file["path"], repo_type="dataset", revision=spec["revision"])
        shutil.copyfile(cached, dest)
        return
    owner_repo = spec["repo"].removeprefix("https://github.com/")
    url = f"https://raw.githubusercontent.com/{owner_repo}/{spec['revision']}/{file['path']}"
    with urllib.request.urlopen(url, timeout=60) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def fetch_files(spec: dict, target: Path, force: bool) -> str:
    fetched = 0
    for file in spec["files"]:
        dest = target / file["to"]
        if not force and dest.exists() and sha256_of(dest) == file["sha256"]:
            continue
        download_file(spec, file, dest)
        if sha256_of(dest) != file["sha256"]:
            return f"FAILED ({file['to']}: sha256 does not match the manifest)"
        fetched += 1
    count = len(spec["files"])
    return f"OK ({fetched} of {count} files fetched)" if fetched else f"SKIP ({count} files match their sha256)"


def git_head(target: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def fetch_checkout(spec: dict, target: Path) -> str:
    revision = spec["revision"]
    if target.exists():
        head = git_head(target)
        if head == revision:
            return f"SKIP (checkout at {revision[:8]})"
        return f"DIFFERENT (the folder is at {(head or 'no git commit')[:8]}, the pin is {revision[:8]}; left as it is)"
    subprocess.run(["git", "clone", "--quiet", spec["repo"], str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--quiet", revision], check=True)
    return f"OK (cloned at {revision[:8]})"


def fetch_external(spec: dict, root: Path, force: bool) -> str:
    target = root / spec["local_dir"]
    try:
        outcome = fetch_files(spec, target, force) if "files" in spec else fetch_checkout(spec, target)
    except (HfHubHTTPError, OSError, subprocess.CalledProcessError) as error:
        return f"FAILED ({type(error).__name__}: {error})"
    if spec.get("after") and outcome.startswith("OK"):
        outcome += f"\n      next: {spec['after']}"
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["train", "eval"], help="fetch one role only")
    parser.add_argument("--only", action="append", default=[], help="fetch a named dataset")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    root, datasets, external = load_manifest()
    by_entry = {e["repo"] for e in external if e["kind"] == "hf"}
    datasets = [d for d in datasets if d["id"] not in by_entry]
    if args.only:
        wanted = set(args.only)
        datasets = [d for d in datasets if d["id"] in wanted]
        external = [e for e in external if e["name"] in wanted]
        missing = wanted - {d["id"] for d in datasets} - {e["name"] for e in external}
        if missing:
            print(f"not in the manifest: {', '.join(sorted(missing))}", file=sys.stderr)
            return 2
    else:
        datasets = [d for d in datasets if not args.role or d.get("role") == args.role]
        external = [e for e in external if not e.get("optional")]

    root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    records = json.loads(INVENTORY.read_text()) if INVENTORY.exists() else {}

    print(f"data root: {root}")
    print(f"datasets : {len(datasets)}, external sources: {len(external)}\n")

    failures = []
    for spec in datasets:
        repo_id = spec["id"]
        print(f"  {repo_id:52s} ", end="", flush=True)
        outcome = fetch_one(api, spec, root, args.force, records)
        print(outcome)
        if outcome.startswith(("FAILED", "MISSING", "HTTP")):
            failures.append(repo_id)
        write_inventory(records)

    for spec in external:
        print(f"  {spec['name']:52s} ", end="", flush=True)
        outcome = fetch_external(spec, root, args.force)
        print(outcome)
        if outcome.startswith(("FAILED", "DIFFERENT")):
            failures.append(spec["name"])

    total = sum(r.get("bytes", 0) for r in records.values())
    print(f"\ntotal on disk: {total / 1e9:.2f} GB")
    print(f"inventory   : {INVENTORY}")
    if failures:
        print(f"failures    : {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
