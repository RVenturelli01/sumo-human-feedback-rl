#!/usr/bin/env python3
"""Fetch the expert demonstrations from Hugging Face.

The repository is public: no login is needed. `--token` is there only for the
case where it is made private again.

Usage:
    python experiments/download_datasets.py
    python experiments/download_datasets.py --out somewhere/else
    python experiments/download_datasets.py --files expert_trajectories.pkl
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "Andrea02/sumo-rlhf-datasets"
#: The commit the checksums in datasets/SHA256SUMS were taken from. Pinned so a
#: later upload to the dataset repository does not silently change the data.
REVISION = "70d5165605c0c2a9bb4d1970c3e1fe24c0ae5f63"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "datasets"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="destination (default: <repo>/datasets)")
    p.add_argument("--files", nargs="*", default=None,
                   help="specific file names (default: every .pkl)")
    p.add_argument("--token", default=None,
                   help="HF token; falls back to the login cache or HF_TOKEN")
    p.add_argument("--revision", default=REVISION,
                   help=f"commit to download (default: {REVISION[:12]})")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID}@{args.revision[:12]} -> {args.out}")
    local = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.out,
        allow_patterns=list(args.files) if args.files else ["*.pkl"],
        token=args.token,
    )

    files = sorted(Path(local).glob("*.pkl"))
    print(f"\n{len(files)} file(s) in {local}:")
    for f in files:
        print(f"  {f.name:44s} {f.stat().st_size / 1e6:7.1f} MB")


if __name__ == "__main__":
    main()
