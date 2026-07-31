#!/usr/bin/env python3
"""Ri-disegna le curve ESISTENTI su un asse di feedback appaiato.

Nessun run nuovo: prende i run gia' finiti dai progetti W&B

    tuning-thesis-budget-curves-completion   (spina dorsale: tutti i bracci, tutti i livelli)
    thesis                                   (solo i due bracci di sole preferenze, budget non ambiguo)

e li riposiziona sull'asse

    B = eventi equivalenti di feedback = query di preferenza + transizioni esperte

Perche' serve
-------------
Le curve originali mettono sullo stesso x "2723 preferenze" e "2723 traiettorie
esperte". Ma 2723 traiettorie sono 487.782 transizioni: sull'asse appaiato quel
punto si sposta da x=2723 a x~490.000, cioe' di oltre due ordini di grandezza.
E' lo spostamento che rende il confronto onesto.

Il numero di transizioni per un budget in traiettorie e' DETERMINISTICO dato il
demo_subsample_seed (prefisso di una permutazione seminata), quindi si calcola
esattamente dal dataset invece di dipendere dai log.

Bracci inclusi, e perche'
-------------------------
  pref_soft, pref_bernoulli      baseline di sole preferenze
  demo_1, demo_2                 baseline di sole dimostrazioni (entrambe normalize=true)
  hybrid_demo_2_soft_hom         ibrido CORRETTO (temperatura allineata, config tunata)
  hybrid_demo_2_bern_hom         ibrido CORRETTO
  hybrid_demo_2_soft_trmatch     punto appaiato per transizioni

Esclusi deliberatamente
-----------------------
  hybrid_demo_2_soft / _bernoulli   versioni PRE-FIX: giravano con i parametri
                                    tunati scartati (demo_weight forzato a 1.0)
                                    e, per il bernoulli, con pref_temperature
                                    25,1 contro il 3,06 della sua baseline.
  demo_2_no_norm                    normalize=false: unico braccio su un asse
                                    di variazione diverso, e nessun livello in
                                    comune con demo_2 (1/10/20 contro 50..2723).
  hybrid_demo_1_*, *_A/_B da thesis run finali a budget singolo con le config
                                    ibride vecchie.

Uso:
    python scripts/retro_unit_axis.py            # tabella + figura
    python scripts/retro_unit_axis.py --table    # solo tabella
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402

ENTITY = "andrea02polimi-politecnico-di-milano"
CURVES_PROJECT = "tuning-thesis-budget-curves-completion"
THESIS_PROJECT = "thesis"
OUT_DIR = REPO_ROOT / "outputs" / "unit_matched_curves"

RETURN_METRIC = "sweep/mean_fast_return"
SUCCESS_METRIC = "sweep/success_rate"

# arm -> (usa preferenze, usa demo, etichetta, stile)
ARMS = {
    "pref_soft":                  (True,  False, "pref_soft",              "-"),
    "pref_bernoulli":             (True,  False, "pref_bernoulli",         "-"),
    "demo_1":                     (False, True,  "demo_1",                 "-"),
    "demo_2":                     (False, True,  "demo_2",                 "-"),
    "hybrid_demo_2_soft_hom":     (True,  True,  "hybrid_soft (corretto)", "-"),
    "hybrid_demo_2_bern_hom":     (True,  True,  "hybrid_bern (corretto)", "-"),
    "hybrid_demo_2_soft_trmatch": (True,  True,  "hybrid_soft trmatch",    "--"),
}
# Dai run finali del progetto thesis: solo sole-preferenze, budget non ambiguo.
THESIS_GROUPS = {
    "pref_soft": ("pref_soft", 5000),
    "pref_bernoulli_q100k_temp": ("pref_bernoulli", 100000),
}

GROUP_RE = re.compile(r"^budget_(.+)_(\d+)$")


# --- mappa traiettorie -> transizioni, esatta ---------------------------------

_LENGTHS: np.ndarray | None = None


def trajectory_lengths() -> np.ndarray:
    global _LENGTHS
    if _LENGTHS is None:
        from _common import load_expert_trajectories
        trajectories = load_expert_trajectories()
        _LENGTHS = np.array([len(t) for t in trajectories])
    return _LENGTHS


_CACHE: dict[tuple[int, int], int] = {}


def transitions_for(n_trajectories: int, subsample_seed: int) -> int:
    """Transizioni consumate dal prefisso di n traiettorie della permutazione seminata.

    Identico a _common.load_expert_trajectories: order[:n] della permutazione
    di default_rng(seed). Deterministico, quindi ricostruibile a posteriori.
    """
    key = (n_trajectories, subsample_seed)
    if key not in _CACHE:
        lengths = trajectory_lengths()
        order = np.random.default_rng(subsample_seed).permutation(len(lengths))
        _CACHE[key] = int(lengths[order[:n_trajectories]].sum())
    return _CACHE[key]


def budget_for(arm: str, level: int, seed: int) -> tuple[int, int, int]:
    """(B totale, query di preferenza, transizioni esperte) per un run."""
    uses_pref, uses_demo, _, _ = ARMS[arm]
    if arm == "hybrid_demo_2_soft_trmatch":
        # 500 query + tetto 500 transizioni (traiettorie intere sotto il tetto)
        lengths = trajectory_lengths()
        order = np.random.default_rng(1000 + seed).permutation(len(lengths))
        total = 0
        for index in order:
            length = int(lengths[index])
            if total and total + length > 500:
                break
            total += length
        return 500 + total, 500, total
    if uses_pref and uses_demo:
        half = level // 2
        transitions = transitions_for(half, 1000 + seed)
        return half + transitions, half, transitions
    if uses_pref:
        return level, level, 0
    transitions = transitions_for(level, 1000 + seed)
    return transitions, 0, transitions


# --- raccolta ----------------------------------------------------------------

def seed_of(run) -> int:
    match = re.search(r"seed(\d+)", run.name or "")
    if match:
        return int(match.group(1))
    return 1


def collect() -> dict:
    import wandb
    api = wandb.Api(timeout=60)
    per_point: dict[tuple[str, int], dict] = defaultdict(
        lambda: {"B": [], "ret": [], "suc": [], "pref": [], "trans": []}
    )

    for run in api.runs(f"{ENTITY}/{CURVES_PROJECT}", per_page=400):
        if run.state != "finished":
            continue
        match = GROUP_RE.match(run.group or "")
        if not match:
            continue
        arm, level = match.group(1), int(match.group(2))
        if arm not in ARMS:
            continue
        value = run.summary.get(RETURN_METRIC)
        if value is None:
            continue
        B, pref, trans = budget_for(arm, level, seed_of(run))
        entry = per_point[(arm, level)]
        entry["B"].append(B)
        entry["ret"].append(float(value))
        entry["pref"].append(pref)
        entry["trans"].append(trans)
        if run.summary.get(SUCCESS_METRIC) is not None:
            entry["suc"].append(float(run.summary[SUCCESS_METRIC]))

    for run in api.runs(f"{ENTITY}/{THESIS_PROJECT}", per_page=400):
        if run.state != "finished" or (run.group or "") not in THESIS_GROUPS:
            continue
        arm, level = THESIS_GROUPS[run.group]
        value = run.summary.get(RETURN_METRIC)
        if value is None:
            continue
        entry = per_point[(arm, level)]
        entry["B"].append(level)
        entry["ret"].append(float(value))
        entry["pref"].append(level)
        entry["trans"].append(0)
        if run.summary.get(SUCCESS_METRIC) is not None:
            entry["suc"].append(float(run.summary[SUCCESS_METRIC]))
    return per_point


def summarise(per_point: dict) -> list[dict]:
    rows = []
    for (arm, level), entry in per_point.items():
        rows.append({
            "arm": arm, "level": level, "n": len(entry["ret"]),
            "B": statistics.mean(entry["B"]),
            "B_min": min(entry["B"]), "B_max": max(entry["B"]),
            "pref": statistics.mean(entry["pref"]),
            "trans": statistics.mean(entry["trans"]),
            "ret": statistics.mean(entry["ret"]),
            "ret_std": statistics.pstdev(entry["ret"]) if len(entry["ret"]) > 1 else 0.0,
            "suc": statistics.mean(entry["suc"]) if entry["suc"] else None,
            "suc_std": (statistics.pstdev(entry["suc"])
                        if len(entry["suc"]) > 1 else 0.0),
        })
    return sorted(rows, key=lambda r: (r["arm"], r["B"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table", action="store_true")
    args = parser.parse_args()

    rows = summarise(collect())

    print("=== curve esistenti riposizionate sull'asse appaiato ===")
    print("B = query di preferenza + transizioni esperte (eventi equivalenti di feedback)")
    print()
    header = ("%-26s %-7s %3s %10s %8s %9s %8s %7s"
              % ("braccio", "livello", "n", "B", "query", "transiz.", "return", "std"))
    print(header)
    print("-" * len(header))
    last = None
    for row in rows:
        if row["arm"] != last:
            print()
            last = row["arm"]
        print("%-26s %-7d %3d %10.0f %8.0f %9.0f %8.2f %7.2f" % (
            row["arm"], row["level"], row["n"], row["B"], row["pref"],
            row["trans"], row["ret"], row["ret_std"]))

    print()
    print("=== spostamento sull'asse: livello nominale -> B reale ===")
    for row in rows:
        if row["trans"]:
            print("  %-26s livello %-6d -> B %8.0f  (fattore %.0fx)" % (
                row["arm"], row["level"], row["B"], row["B"] / max(row["level"], 1)))

    if args.table:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # noqa: BLE001
        print(f"\n(matplotlib non disponibile: {error})")
        return

    by_arm = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for metric, ax, title, ylabel in (
        ("ret", axes[0], "Held-out mean fast return", "mean_fast_return"),
        ("suc", axes[1], "Held-out success rate", "success_rate"),
    ):
        for arm, entries in by_arm.items():
            _, _, label, style = ARMS[arm]
            xs = [e["B"] for e in entries]
            ys = [e[metric] for e in entries if e[metric] is not None]
            es = [e[f"{metric}_std"] for e in entries if e[metric] is not None]
            if len(ys) != len(xs):
                xs = [e["B"] for e in entries if e[metric] is not None]
            if not xs:
                continue
            ax.errorbar(xs, ys, yerr=es, marker="o", markersize=5, capsize=3,
                        linewidth=1.8, linestyle=style, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("B · eventi equivalenti di feedback (query + transizioni esperte)")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25)
        ax.set_ylabel(ylabel)
    axes[1].set_ylim(-0.02, 1.02)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Curve esistenti riposizionate sull'asse di feedback appaiato — 2M step, "
        "media ± std sui seed", fontsize=11)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = OUT_DIR / f"retro_unit_axis.{suffix}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\nfigura: {path}")


if __name__ == "__main__":
    main()
