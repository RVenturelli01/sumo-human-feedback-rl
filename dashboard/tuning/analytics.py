"""Small, explicit diagnostics for the tuning dashboard."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


def scored(
    records: list[dict[str, Any]], algorithm: str
) -> list[dict[str, Any]]:
    return sorted(
        [
            row
            for row in records
            if row["algorithm"] == algorithm and row["score"] is not None
        ],
        key=lambda row: row["trial"],
    )


def all_params(rows: list[dict[str, Any]]) -> list[str]:
    names = {name for row in rows for name in row.get("params", {})}
    return sorted(names)


def display_value(value: Any) -> str:
    if isinstance(value, list):
        return " x ".join(str(item) for item in value)
    if isinstance(value, float):
        return f"{value:.5g}"
    return str(value)


def _numeric(values: list[Any]) -> bool:
    return bool(values) and all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in values
    )


def distance_matrix(
    rows: list[dict[str, Any]], top_k: int
) -> tuple[list[str], np.ndarray, list[dict[str, Any]]]:
    ranked = sorted(rows, key=lambda row: row["score"], reverse=True)[:top_k]
    names = all_params(rows)
    matrix = np.zeros((len(ranked), len(ranked)), dtype=float)

    ranges: dict[str, tuple[float, float, bool]] = {}
    for name in names:
        values = [
            row["params"][name]
            for row in rows
            if name in row.get("params", {})
        ]
        if not _numeric(values):
            continue
        numeric_values = [float(value) for value in values]
        positive = min(numeric_values) > 0
        log_scale = (
            positive and max(numeric_values) / min(numeric_values) >= 100
        )
        transformed = (
            [math.log10(value) for value in numeric_values]
            if log_scale
            else numeric_values
        )
        ranges[name] = (min(transformed), max(transformed), log_scale)

    for left, right in combinations(range(len(ranked)), 2):
        parts: list[float] = []
        left_params = ranked[left].get("params", {})
        right_params = ranked[right].get("params", {})
        for name in names:
            if name not in left_params or name not in right_params:
                continue
            a, b = left_params[name], right_params[name]
            if name in ranges:
                low, high, log_scale = ranges[name]
                af, bf = float(a), float(b)
                if log_scale:
                    af, bf = math.log10(af), math.log10(bf)
                parts.append(abs(af - bf) / (high - low) if high > low else 0.0)
            else:
                parts.append(0.0 if display_value(a) == display_value(b) else 1.0)
        distance = float(np.mean(parts)) if parts else 0.0
        matrix[left, right] = matrix[right, left] = distance

    labels = [f"t{row['trial']:03d}" for row in ranked]
    return labels, matrix, ranked


def isolation_ratio(
    rows: list[dict[str, Any]], top_k: int = 5
) -> float | None:
    _, matrix, ranked = distance_matrix(rows, top_k)
    if len(ranked) < 3:
        return None
    best_distance = float(np.mean(matrix[0, 1:]))
    other_distances = [
        matrix[i, j]
        for i in range(1, len(ranked))
        for j in range(i + 1, len(ranked))
    ]
    baseline = float(np.median(other_distances)) if other_distances else 0.0
    return best_distance / baseline if baseline > 0 else None


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    ranked = sorted(rows, key=lambda row: row["score"], reverse=True)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    max_trial = max(row["trial"] for row in rows)
    previous = [
        row["score"] for row in rows if row["trial"] <= max_trial - 10
    ]
    late_gain = best["score"] - max(previous) if previous else None
    within_one = sum(best["score"] - row["score"] <= 1.0 for row in rows)
    within_two = sum(best["score"] - row["score"] <= 2.0 for row in rows)
    isolation = isolation_ratio(rows)

    if within_one >= 3 and late_gain is not None and late_gain <= 1.0:
        diagnosis = "Plateau probabile"
        tone = "stable"
    elif (
        isolation is not None
        and isolation >= 1.25
        and second
        and best["score"] - second["score"] < 1.0
    ):
        diagnosis = "Candidato isolato"
        tone = "warning"
    elif best["trial"] >= 20 and late_gain is not None and late_gain > 0.75:
        diagnosis = "Ancora in miglioramento"
        tone = "active"
    else:
        diagnosis = "Da verificare"
        tone = "neutral"

    return {
        "best": best,
        "gap": best["score"] - second["score"] if second else None,
        "second": second,
        "within_one": within_one,
        "within_two": within_two,
        "late_gain": late_gain,
        "isolation": isolation,
        "diagnosis": diagnosis,
        "tone": tone,
    }


def parameter_association(
    rows: list[dict[str, Any]],
) -> list[tuple[str, float]]:
    """Return univariate association strength, not causal importance."""
    result: list[tuple[str, float]] = []
    scores = np.array([row["score"] for row in rows], dtype=float)

    for name in all_params(rows):
        pairs = [
            (row["params"].get(name), row["score"])
            for row in rows
            if name in row.get("params", {})
        ]
        values = [value for value, _ in pairs]
        if len(pairs) < 5 or len({display_value(value) for value in values}) < 2:
            continue
        if _numeric(values):
            frame = pd.DataFrame(
                {
                    "x": [float(value) for value in values],
                    "y": [score for _, score in pairs],
                }
            )
            if (
                frame["x"].min() > 0
                and frame["x"].max() / frame["x"].min() >= 100
            ):
                frame["x"] = np.log10(frame["x"])
            strength = abs(
                float(frame["x"].corr(frame["y"], method="spearman"))
            )
        else:
            frame = pd.DataFrame(
                {
                    "x": [display_value(value) for value in values],
                    "y": [score for _, score in pairs],
                }
            )
            overall = frame["y"].mean()
            between = sum(
                len(group) * (group["y"].mean() - overall) ** 2
                for _, group in frame.groupby("x")
            )
            denominator = float(np.sum((frame["y"] - overall) ** 2))
            strength = float(between / denominator) if denominator > 0 else 0.0
        if not math.isnan(strength):
            result.append((name, strength))
    return sorted(result, key=lambda item: item[1], reverse=True)


def candidate_table(
    rows: list[dict[str, Any]], top_k: int
) -> tuple[list[dict[str, Any]], list[str]]:
    ranked = sorted(rows, key=lambda row: row["score"], reverse=True)[:top_k]
    params = all_params(ranked)
    data = []
    for rank, row in enumerate(ranked, 1):
        item: dict[str, Any] = {
            "rank": rank,
            "trial": f"t{row['trial']:03d}",
            "score": round(row["score"], 4),
        }
        item.update(
            {
                name: display_value(row.get("params", {}).get(name, "-"))
                for name in params
            }
        )
        data.append(item)
    return data, ["rank", "trial", "score", *params]
