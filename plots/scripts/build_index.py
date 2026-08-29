#!/usr/bin/env python
"""Build or refresh the cache of W&B run metadata.

    python plots/scripts/build_index.py            # incremental update
    python plots/scripts/build_index.py --force    # refetch every config
    python plots/scripts/build_index.py --projects thesis-final
"""
import argparse

import _bootstrap  # noqa: F401
from rtplots.index import build_index
from rtplots.paths import DEFAULT_PROJECTS


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true", help="ignora la cache e riscarica tutto")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--projects", nargs="+", default=DEFAULT_PROJECTS,
                   help=f"progetti W&B da indicizzare (default: {' '.join(DEFAULT_PROJECTS)})")
    args = p.parse_args()

    df = build_index(force=args.force, workers=args.workers, projects=args.projects)

    print("\n[index] run per arm:")
    print(df.arm.value_counts(dropna=False).to_string())
    print("\n[index] run per stato:")
    print(df.state.value_counts(dropna=False).to_string())
    if "error" in df.columns:
        broken = df[df.error.notna()] if "error" in df else df.iloc[0:0]
        if len(broken):
            print(f"\n[index] ATTENZIONE: {len(broken)} run non lette correttamente "
                  f"(see the 'error' column)")


if __name__ == "__main__":
    main()
