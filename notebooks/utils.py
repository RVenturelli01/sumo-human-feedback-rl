import json as _json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


def _safe_get(d, *keys, default=np.nan):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def load_runs(entity: str, project: str, cache: Path, force_reload: bool = False) -> pd.DataFrame:
    if cache.exists() and not force_reload:
        print(f"Caricato da cache: {cache}")
        return pd.read_csv(cache)

    print("Download runs da WandB API...")
    api  = wandb.Api(timeout=60)
    runs = api.runs(f"{entity}/{project}")

    states = Counter(r.state for r in runs)
    print("Stato runs:")
    for state, n in sorted(states.items()):
        print(f"  {state:15s}: {n}")
    print(f"  {'TOTAL':15s}: {sum(states.values())}")

    rows = []
    for r in runs:
        r.load_full_data()
        cfg = r.config
        s   = r.summary._json_dict
        row = {
            "run_id"              : r.id,
            "run_name"            : r.name,
            "state"               : r.state,
            "lr_rew"              : _safe_get(cfg, "algo", "kwargs", "lr_rew"),
            "gradient_steps_rew"  : _safe_get(cfg, "algo", "kwargs", "gradient_steps_rew"),
            "batch_size_rew"      : _safe_get(cfg, "algo", "kwargs", "batch_size_rew"),
            "fragment_length"     : _safe_get(cfg, "algo", "kwargs", "fragment_length"),
            "fragmenter_type"     : _safe_get(cfg, "algo", "kwargs", "fragmenter_type"),
            "labels_type"         : _safe_get(cfg, "algo", "kwargs", "labels_type"),
            "net_arch"            : str(_safe_get(cfg, "algo", "kwargs", "reward_model_kwargs", "net_arch")),
            "total_queries"       : _safe_get(cfg, "train", "kwargs", "total_queries"),
            "successes"           : s.get("agent/event_rate/successes",      np.nan),
            "timeouts"            : s.get("agent/event_rate/timeouts",       np.nan),
            "collisions"          : s.get("agent/event_rate/collisions",     np.nan),
            "ep_length"           : s.get("agent/performance/ep_length",     np.nan),
            "ep_avg_speed"        : s.get("agent/performance/ep_avg_speed",  np.nan),
            "ep_env_return"       : s.get("agent/rewards/ep_env_return",     np.nan),
            "ep_fast_return"      : s.get("agent/rewards/ep_fast_return",    np.nan),
            "spearman_seg1"       : s.get("reward_correlation/spearman_seg1",    np.nan),
            "spearman_seg5"       : s.get("reward_correlation/spearman_seg5",    np.nan),
            "spearman_seg20"      : s.get("reward_correlation/spearman_seg20",   np.nan),
            "spearman_segNone"    : s.get("reward_correlation/spearman_segNone", np.nan),
            "acc_val"             : s.get("reward/acc_all_val",                  np.nan),
            "acc_train"           : s.get("reward/acc_all_train",                np.nan),
            "acc_gap"             : s.get("reward/acc_all_gap",                  np.nan),
            "loss_val"            : s.get("reward/loss_all_val",                 np.nan),
            "mean_true_reward"    : s.get("rollout/mean_true_reward",            np.nan),
            "mean_model_reward"   : s.get("rollout/mean_model_reward",           np.nan),
            "time_train_rew_model": s.get("time/train_reward_model",             np.nan),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    print(f"Salvato in {cache}")
    return df


def encode_hp(sub_df: pd.DataFrame, hp_cols: list) -> pd.DataFrame:
    X = sub_df[hp_cols].copy()
    for col in ["fragmenter_type", "labels_type", "net_arch"]:
        if col in X.columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X["fragment_length"] = (
        X["fragment_length"].replace("None", -1)
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(-1)
    )
    return X


def rf_importance(df: pd.DataFrame, hp_cols: list, target_metrics: dict) -> pd.DataFrame:
    """Rows = hp_cols, columns = metric labels. Values = RF feature importances."""
    result = {}
    for label, col in target_metrics.items():
        if col not in df.columns:
            continue
        sub = df[hp_cols + [col]].dropna(subset=[col]).copy()
        if len(sub) < 10:
            continue
        X = encode_hp(sub, hp_cols)
        y = sub[col].values
        rf = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
        rf.fit(X, y)
        result[label] = pd.Series(rf.feature_importances_, index=hp_cols)
    return pd.DataFrame(result)
