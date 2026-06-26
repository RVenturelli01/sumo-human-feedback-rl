#!/usr/bin/env python3
"""Scarica i dataset Parquet SUMO-RLHF dal repo privato su Hugging Face.

Repo: https://huggingface.co/datasets/Andrea02/sumo-rlhf-datasets (privato)

Autenticazione (necessaria perche' il dataset e' privato), scegli una via:
  - `hf auth login`               (token salvato in locale)
  - export HF_TOKEN=hf_xxx        (variabile d'ambiente)

Uso:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --out data_for_training/parquet
    python scripts/download_datasets.py --files expert_trajectories.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "Andrea02/sumo-rlhf-datasets"
REPO_TYPE = "dataset"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data_for_training" / "parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Cartella di destinazione (default: data_for_training/parquet).",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Nomi di file specifici da scaricare (default: tutti i .parquet).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token; se omesso usa `hf auth login` o la env var HF_TOKEN.",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    patterns = list(args.files) if args.files else ["*.parquet"]

    print(f"Scarico {REPO_ID} -> {args.out}")
    local_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=args.out,
        allow_patterns=patterns,
        token=args.token,  # None => usa cache login / HF_TOKEN
    )

    downloaded = sorted(Path(local_path).glob("*.parquet"))
    print(f"\nFatto. {len(downloaded)} file in {local_path}:")
    for f in downloaded:
        print(f"  {f.name:40s} {f.stat().st_size / 1e6:7.1f} MB")


if __name__ == "__main__":
    main()
