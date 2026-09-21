"""Download and unpack the SciFact dataset.

Fetches the official AllenAI SciFact release tarball and extracts it into
`data/raw/`. Idempotent: skips download if the target directory already
has the expected files.

Usage:
    python -m data.download_data
    python -m data.download_data --config configs/data.yaml

Network note:
    This script requires outbound access to scifact.s3-us-west-2.amazonaws.com.
    Some sandboxed / firewalled environments (including the one this project
    was drafted in) whitelist only a small set of domains and will reject
    this request with a 403. If that happens here, run this script on an
    unrestricted machine (your laptop, a Colab notebook, CI, etc.) instead.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

import yaml

EXPECTED_FILES = [
    "corpus.jsonl",
    "claims_train.jsonl",
    "claims_dev.jsonl",
    "claims_test.jsonl",
]


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def already_downloaded(raw_dir: Path) -> bool:
    return all((raw_dir / fname).exists() for fname in EXPECTED_FILES)


def download_and_extract(source_url: str, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    tarball_path = raw_dir / "data.tar.gz"

    print(f"Downloading {source_url} -> {tarball_path}")
    try:
        urllib.request.urlretrieve(source_url, tarball_path)
    except Exception as exc:  # noqa: BLE001 - surface network errors clearly
        raise RuntimeError(
            f"Failed to download SciFact data from {source_url}. "
            "If you're behind a restrictive network/firewall (e.g. a sandboxed "
            "dev environment), run this script on an unrestricted machine "
            f"instead. Original error: {exc}"
        ) from exc

    print("Extracting...")
    with tarfile.open(tarball_path, "r:gz") as tar:
        tar.extractall(path=raw_dir.parent)  # tarball unpacks its own "data/" dir

    tarball_path.unlink(missing_ok=True)

    # The AllenAI tarball extracts to a top-level "data/" folder. Reconcile
    # that with our configured raw_dir if names differ.
    extracted = raw_dir.parent / "data"
    if extracted != raw_dir and extracted.exists():
        for item in extracted.iterdir():
            shutil.move(str(item), str(raw_dir / item.name))
        extracted.rmdir()


def verify(raw_dir: Path) -> dict:
    """Sanity-check the download and return basic counts (no fabricated stats)."""
    counts = {}
    for fname in EXPECTED_FILES:
        fpath = raw_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Expected file missing after download: {fpath}")
        with open(fpath, "r", encoding="utf-8") as f:
            counts[fname] = sum(1 for _ in f)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = Path(config["paths"]["raw_dir"])
    source_url = config["dataset"]["source_url"]

    if not args.force and already_downloaded(raw_dir):
        print(f"Data already present in {raw_dir}. Use --force to re-download.")
    else:
        download_and_extract(source_url, raw_dir)

    counts = verify(raw_dir)
    print("\nDownload verified. Line counts per file:")
    for fname, n in counts.items():
        print(f"  {fname}: {n}")
    print(
        "\nNote: line counts above are the ground truth for this run — "
        "they are not hardcoded and will be used verbatim in the research report."
    )


if __name__ == "__main__":
    sys.exit(main())
