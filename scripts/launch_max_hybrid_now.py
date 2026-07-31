#!/usr/bin/env python3
"""Launch the approved 2723-pref + 2723-demo hybrid triplet."""

from __future__ import annotations

import json

from scripts import schedule_budget_curves_completion as budget_runner


def main() -> None:
    tasks = budget_runner.build_tasks()
    state = budget_runner.load_transition_state(tasks)
    selected = tuple(
        task
        for task in tasks
        if task.arm == "hybrid_demo_2_bernoulli"
        and task.point.pref_budget == 2_723
        and task.point.demo_budget == 2_723
    )
    if len(selected) != 3:
        raise RuntimeError("Expected exactly three max-budget hybrid tasks.")

    processes = budget_runner.iter_processes()
    live_names = set(budget_runner.live_training_runs(processes))
    for task in selected:
        if (
            task.run_name in live_names
            or budget_runner.read_marker(task)
            or task.output_root.exists()
            or task.log_path.exists()
        ):
            raise RuntimeError(
                f"Refusing to duplicate or resume {task.run_name}."
            )

    free_slots = [
        slot
        for slot in budget_runner.SINGLE_SLOTS
        if not budget_runner.constrained_slot_processes(processes, slot)
    ]
    if len(free_slots) < len(selected):
        raise RuntimeError(
            f"Only {len(free_slots)} singleton slots are free."
        )

    scheduler = budget_runner.Scheduler(
        tasks,
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
                "pref_budget": task.point.pref_budget,
                "demo_budget": task.point.demo_budget,
                "initial_queries": task.point.initial_queries,
                "seed": task.seed,
            }
        )
    print(json.dumps(launched, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
