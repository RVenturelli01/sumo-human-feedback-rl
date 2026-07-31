#!/usr/bin/env python3
"""Resolve and validate the two preference-Bernoulli budget configs."""

from __future__ import annotations

import json

from scripts import schedule_budget_curves_completion as budget_runner


def main() -> None:
    tasks = tuple(
        task
        for task in budget_runner.build_tasks()
        if task.arm == "pref_bernoulli"
    )
    records = budget_runner.validate_resolved_configs(tasks)
    print(json.dumps(records, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
