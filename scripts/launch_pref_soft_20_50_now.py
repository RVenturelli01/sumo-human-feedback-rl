#!/usr/bin/env python3
"""Launch pref-soft budgets 20 and 50 for seeds 1-3."""

from __future__ import annotations

import json

from scripts import schedule_budget_curves_completion as budget_runner


def main() -> None:
    points = tuple(
        budget_runner.Point(
            arm="pref_soft",
            budget=budget,
            source_key="pref_soft",
            pref_budget=budget,
            demo_budget=None,
            initial_queries=budget // 5,
            normalize_agent_reward=True,
            labels_type="soft",
            loss_type="demo_2",
            query_schedule="constant",
            fragmenter_type="random",
            pref_temperature=20.0,
            demo_weight=0.0,
        )
        for budget in (20, 50)
    )
    selected = tuple(
        budget_runner.Task(point, seed)
        for point in points
        for seed in budget_runner.SEEDS
    )
    matrix_tasks = budget_runner.build_tasks()
    state = budget_runner.load_transition_state(matrix_tasks)
    processes = budget_runner.iter_processes()
    live_names = set(budget_runner.live_training_runs(processes))

    for task in selected:
        if (
            task.run_name in live_names
            or budget_runner.read_marker(task)
            or task.output_root.exists()
            or task.log_path.exists()
        ):
            raise RuntimeError(f"Refusing to duplicate {task.run_name}.")

    free_slots = [
        slot
        for slot in budget_runner.SINGLE_SLOTS
        if not budget_runner.constrained_slot_processes(processes, slot)
    ]
    if len(free_slots) < len(selected):
        raise RuntimeError(f"Only {len(free_slots)} singleton slots are free.")

    scheduler = budget_runner.Scheduler(
        matrix_tasks,
        max_parallel=24,
        max_attempts=budget_runner.MAX_ATTEMPTS,
        loop_seconds=budget_runner.LOOP_SECONDS,
        transition_state=state,
    )
    launched = []
    for task, slot in zip(selected, free_slots):
        if not scheduler.launch(task, slot, budget_runner.iter_processes()):
            raise RuntimeError(f"Slot {slot} became occupied before launch.")
        running = scheduler.running[task.key]
        launched.append(
            {
                "run_name": task.run_name,
                "pid": running.pid,
                "slot": slot,
                "budget": task.budget,
                "initial_queries": task.point.initial_queries,
                "seed": task.seed,
            }
        )
    print(json.dumps(launched, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
