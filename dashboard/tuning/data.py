"""Load tuning results from W&B and parameters from the Optuna journal."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import optuna
import wandb
from optuna.importance import FanovaImportanceEvaluator, get_param_importances
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


ROOT = Path(__file__).resolve().parent
CACHE_FILE = ROOT / "cache" / "tuning_runs.json"
PROJECT = os.getenv(
    "TUNING_WANDB_PROJECT",
    "andrea02polimi-politecnico-di-milano/tuning-thesis",
)
JOURNAL = Path(
    os.getenv(
        "TUNING_OPTUNA_JOURNAL",
        "/work/fis3/sumo-human-feedback-rl/outputs/optuna/journal.log",
    )
)
SCORE_KEY = "sweep/mean_fast_return"

ALGORITHMS: dict[str, dict[str, str]] = {
    "pref_soft": {
        "label": "Soft preference",
        "group": "tune_pref_soft",
        "study": "hybrid_sac_pref_soft",
        "color": "#2f73c9",
    },
    "pref_bernoulli": {
        "label": "Bernoulli preference",
        "group": "tune_pref_bernoulli_q100k_temp",
        "study": "hybrid_sac_pref_bernoulli_q100k_temp",
        "color": "#1e9f79",
    },
    "demo_1": {
        "label": "Demo 1",
        "group": "tune_demo_1",
        "study": "hybrid_sac_demo_1",
        "color": "#e49000",
    },
    "demo_2_no_norm": {
        "label": "Demo 2 (no norm)",
        "group": "tune_demo_2_no_norm",
        "study": "hybrid_sac_demo_2_no_norm",
        "color": "#087d30",
    },
    "hybrid_soft": {
        "label": "Hybrid soft",
        "group": "tune_hybrid_demo_2_hom_soft",
        "study": "hybrid_sac_hybrid_demo_2_hom_soft",
        "color": "#e64f45",
    },
    "hybrid_bernoulli": {
        "label": "Hybrid Bernoulli",
        "group": "tune_hybrid_demo_2_hom_bern",
        "study": "hybrid_sac_hybrid_demo_2_hom_bern",
        "color": "#4b3fb3",
    },
}


def _trial_number(run_name: str) -> int | None:
    match = re.search(r"-t(\d+)$", run_name or "")
    return int(match.group(1)) if match else None


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _load_optuna_data() -> tuple[
    dict[tuple[str, int], dict[str, Any]], dict[str, dict[str, Any]]
]:
    if not JOURNAL.exists():
        raise FileNotFoundError(f"Optuna journal not found: {JOURNAL}")

    storage = JournalStorage(JournalFileBackend(str(JOURNAL)))
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    fanova: dict[str, dict[str, Any]] = {}
    for key, spec in ALGORITHMS.items():
        study = optuna.load_study(study_name=spec["study"], storage=storage)
        for trial in study.trials:
            indexed[(key, trial.number)] = {
                "params": {
                    name: _json_value(value) for name, value in trial.params.items()
                },
                "optuna_state": trial.state.name.lower(),
                "optuna_value": trial.value,
            }
        completed = sum(
            trial.state == optuna.trial.TrialState.COMPLETE
            for trial in study.trials
        )
        try:
            importances = get_param_importances(
                study,
                evaluator=FanovaImportanceEvaluator(seed=0),
            )
            fanova[key] = {
                "completed_trials": completed,
                "importances": {
                    name: float(value) for name, value in importances.items()
                },
                "error": None,
            }
        except Exception as error:
            fanova[key] = {
                "completed_trials": completed,
                "importances": {},
                "error": f"{type(error).__name__}: {error}",
            }
    return indexed, fanova


def _wandb_runs() -> list[Any]:
    api = wandb.Api(timeout=120)
    api.flush()
    groups = [spec["group"] for spec in ALGORITHMS.values()]
    try:
        return list(
            api.runs(
                PROJECT,
                filters={"group": {"$in": groups}},
                per_page=100,
                lazy=False,
            )
        )
    except Exception:
        runs: list[Any] = []
        for group in groups:
            runs.extend(
                api.runs(
                    PROJECT,
                    filters={"group": group},
                    per_page=50,
                    lazy=False,
                )
            )
        return runs


def sync_data() -> dict[str, Any]:
    """Refresh the local cache atomically and return its payload."""
    group_to_key = {spec["group"]: key for key, spec in ALGORITHMS.items()}
    optuna_trials, fanova = _load_optuna_data()
    records: list[dict[str, Any]] = []

    for run in _wandb_runs():
        algorithm = group_to_key.get(run.group)
        trial = _trial_number(run.name)
        if algorithm is None or trial is None:
            continue

        optuna_trial = optuna_trials.get((algorithm, trial), {})
        summary = dict(run.summary)
        wandb_score = summary.get(SCORE_KEY)
        score = wandb_score
        score_source = "wandb"
        if (
            score is None
            and optuna_trial.get("optuna_state") == "complete"
            and optuna_trial.get("optuna_value") is not None
        ):
            score = optuna_trial["optuna_value"]
            score_source = "optuna"

        records.append(
            {
                "algorithm": algorithm,
                "group": run.group,
                "trial": trial,
                "run_id": run.id,
                "run_name": run.name,
                "wandb_state": run.state,
                "optuna_state": optuna_trial.get("optuna_state", "unknown"),
                "score": float(score) if score is not None else None,
                "score_source": score_source if score is not None else None,
                "runtime_seconds": summary.get("_runtime"),
                "created_at": str(run.created_at) if run.created_at else None,
                "url": f"https://wandb.ai/{PROJECT}/runs/{run.id}",
                "params": optuna_trial.get("params", {}),
            }
        )

    records.sort(key=lambda row: (row["algorithm"], row["trial"]))
    payload = {
        "schema_version": 2,
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "project": PROJECT,
        "journal": str(JOURNAL),
        "score_key": SCORE_KEY,
        "algorithms": ALGORITHMS,
        "fanova": fanova,
        "records": records,
    }
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = CACHE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(CACHE_FILE)
    return payload


def load_cache() -> dict[str, Any]:
    if not CACHE_FILE.exists():
        return {
            "schema_version": 2,
            "synced_at": None,
            "project": PROJECT,
            "journal": str(JOURNAL),
            "score_key": SCORE_KEY,
            "algorithms": ALGORITHMS,
            "fanova": {},
            "records": [],
        }
    return json.loads(CACHE_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    result = sync_data()
    print(f"Synced {len(result['records'])} runs at {result['synced_at']}")
