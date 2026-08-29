"""Selecting and grouping runs from the index.

Filter syntax, one per argument:
    arm=demo_1,demo_2              any of these
    arm_family!=demo               not this
    query_budget>=5000             numeric comparison (>, >=, <, <=)
    normalize_agent_reward=false   booleans
"""
from __future__ import annotations

import re

import pandas as pd

_OPS = ["!=", ">=", "<=", "=", ">", "<"]


def _cast(value: str):
    v = value.strip()
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null", "nan", ""):
        return None
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    if re.fullmatch(r"-?\d*\.\d+([eE][-+]?\d+)?", v):
        return float(v)
    return v


def parse_filter(expr: str):
    for op in _OPS:
        if op in expr:
            key, raw = expr.split(op, 1)
            values = [_cast(x) for x in raw.split(",")] if op in ("=", "!=") else _cast(raw)
            return key.strip(), op, values
    raise ValueError(f"Invalid filter: {expr!r} (use key=value, key!=value, key>=value)")


def apply_filters(df: pd.DataFrame, exprs) -> pd.DataFrame:
    """Apply every filter, all of them at once."""
    out = df
    for expr in exprs or []:
        key, op, values = parse_filter(expr)
        if key not in out.columns:
            raise KeyError(f"Unknown column: {key}. Available: {sorted(out.columns)}")
        col = out[key]
        if op == "=":
            mask = col.isin(values) if None not in values else (col.isin(values) | col.isna())
        elif op == "!=":
            mask = ~col.isin(values)
        elif op == ">":
            mask = col > values
        elif op == ">=":
            mask = col >= values
        elif op == "<":
            mask = col < values
        else:
            mask = col <= values
        out = out[mask.fillna(False)]
    return out


def select_runs(df: pd.DataFrame, filters=None, state: str | None = "finished",
                dropna_cols=()) -> pd.DataFrame:
    """Filter the index, keeping only finished runs by default."""
    out = df
    if state and state != "any":
        out = out[out.state.isin(state.split(","))]
    out = apply_filters(out, filters)
    for c in dropna_cols:
        out = out[out[c].notna()]
    return out.copy()


def coverage(df: pd.DataFrame, by) -> pd.DataFrame:
    """How many seeds each combination has, to see what is available."""
    by = list(by)
    g = df.groupby(by, dropna=False).agg(
        n_seeds=("seed", "nunique"), n_runs=("run_id", "size"),
        seeds=("seed", lambda s: ",".join(str(int(x)) for x in sorted(s.dropna().unique()))),
    )
    return g.reset_index().sort_values(by)
