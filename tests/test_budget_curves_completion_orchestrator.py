from collections import Counter, defaultdict
import json
import time
import zipfile

import pytest

from scripts import schedule_budget_curves_completion as orchestrator


def overrides_dict(task):
    return {
        key: value
        for key, value in (
            override.split("=", 1)
            for override in orchestrator.task_overrides(task)
        )
    }


def tasks_by_arm():
    result = defaultdict(list)
    for task in orchestrator.build_tasks():
        result[task.arm].append(task)
    return result


def write_valid_output(task):
    directory = task.output_root / task.run_name
    directory.mkdir(parents=True)
    (directory / "final_eval.json").write_text(
        json.dumps(
            {
                "eval/mean_fast_return": 42.0,
                "eval/success_rate": 0.75,
            }
        )
    )
    with zipfile.ZipFile(directory / "agent_final.zip", "w") as archive:
        archive.writestr("policy.pth", b"test")
    return directory


class FakeProcess:
    def __init__(self, return_codes):
        self.return_codes = iter(return_codes)

    def poll(self):
        return next(self.return_codes)


def fake_process(
    task,
    pid=1234,
    affinity=(24, 25),
    start_time=456,
    pgid=None,
    sid=None,
    command=None,
):
    pgid = pid if pgid is None else pgid
    sid = pid if sid is None else sid
    return orchestrator.ProcessInfo(
        pid=pid,
        ppid=1,
        pgid=pgid,
        sid=sid,
        start_time=start_time,
        state="R",
        affinity=frozenset(affinity),
        command=command
        or (
            "/python scripts/train_hybrid_sac.py "
            f"run.name={task.run_name}"
        ),
    )


def transition_state(tasks, processes, phase=orchestrator.PHASE_DRAIN):
    members = []
    for task, process in zip(tasks, processes):
        slot = "-".join(str(core) for core in sorted(process.affinity))
        members.append(orchestrator.transition_member(task, process, slot))
    state = {
        "version": orchestrator.TRANSITION_VERSION,
        "matrix_sha256": orchestrator.task_matrix_sha256(tuple(tasks)),
        "phase": phase,
        "captured_at": "test",
        "switched_at": None,
        "legacy_manager": {
            "pid": 999,
            "start_time": 111,
            "command": (
                "/python scripts/schedule_budget_curves_completion.py"
            ),
        },
        "cohort": members,
        "history": [],
    }
    if phase == orchestrator.PHASE_SINGLE:
        state.update(
            {
                "switched_at": "test",
                "rollout": "full",
                "canary_task_key": None,
                "canary_pid": None,
                "canary_started_at_epoch": None,
            }
        )
    return orchestrator.validate_transition_state(state, tuple(tasks))


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(orchestrator, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(orchestrator, "RUN_ROOT", tmp_path / "runs")
    monkeypatch.setattr(orchestrator, "LOG_ROOT", tmp_path / "logs")
    monkeypatch.setattr(orchestrator, "MARKER_ROOT", tmp_path / "markers")
    monkeypatch.setattr(
        orchestrator,
        "TRANSITION_STATE_PATH",
        tmp_path / "state" / "transition_state.json",
    )
    monkeypatch.setattr(
        orchestrator,
        "TRANSITION_MANIFEST_PATH",
        tmp_path / "state" / "transition_manifest.json",
    )
    monkeypatch.setattr(
        orchestrator,
        "LOCK_PATH",
        tmp_path / "state" / "orchestrator.lock",
    )
    monkeypatch.setattr(orchestrator, "STOP_REQUESTED", False)
    return tmp_path


def test_matrix_has_20_points_and_60_unique_runs():
    tasks = orchestrator.build_tasks()
    summary = orchestrator.validate_task_matrix(tasks)

    assert summary["tasks"] == 60
    assert summary["points"] == 20
    assert len({task.key for task in tasks}) == 60
    assert len({task.run_name for task in tasks}) == 60
    assert len({task.wandb_run_id for task in tasks}) == 60
    assert all(task.seed in (1, 2, 3) for task in tasks)


def test_exact_point_matrix_and_seed_counts():
    expected = {
        "demo_1": {1, 10, 20},
        "demo_2_no_norm": {1, 10, 20},
        "pref_soft": {10, 100, 250},
        "pref_bernoulli": {1000, 1_000_000},
        "hybrid_demo_2_soft": {2, 20, 200, 2000},
        "hybrid_demo_2_bernoulli": {2, 20, 200, 2000, 5_446},
    }
    by_arm = tasks_by_arm()

    assert set(by_arm) == set(expected)
    for arm, budgets in expected.items():
        assert {task.budget for task in by_arm[arm]} == budgets
        assert Counter(task.budget for task in by_arm[arm]) == {
            budget: 3 for budget in budgets
        }


def test_hybrid_splits_initial_queries_and_approved_overrides():
    expected_initial = {
        2: 1,
        20: 1,
        200: 10,
        2000: 100,
        5_446: 272,
    }
    for task in orchestrator.build_tasks():
        if not task.arm.startswith("hybrid_"):
            continue
        point = task.point
        assert point.pref_budget == point.budget // 2
        assert point.demo_budget == point.budget // 2
        assert point.initial_queries == expected_initial[point.budget]
        assert point.demo_weight == 1.0
        assert point.normalize_agent_reward is True
        assert point.query_schedule == "constant"
        assert task.demo_subsample_seed == 1000 + task.seed


def test_preference_initial_queries_and_schedules():
    expected = {
        ("pref_soft", 10): (2, "constant", "soft", 20.0),
        ("pref_soft", 100): (20, "constant", "soft", 20.0),
        ("pref_soft", 250): (50, "constant", "soft", 20.0),
        ("pref_bernoulli", 1000): (
            200,
            "hyperbolic",
            "binary_bernoulli",
            3.0595414013726767,
        ),
        ("pref_bernoulli", 1_000_000): (
            200_000,
            "hyperbolic",
            "binary_bernoulli",
            3.0595414013726767,
        ),
    }
    for task in orchestrator.build_tasks():
        key = (task.arm, task.budget)
        if key not in expected:
            continue
        initial, schedule, labels, temperature = expected[key]
        assert task.point.initial_queries == initial
        assert task.point.query_schedule == schedule
        assert task.point.labels_type == labels
        assert task.point.pref_temperature == temperature
        assert task.demo_subsample_seed is None


def test_demo_configuration_and_paired_subsample_seeds():
    by_arm = tasks_by_arm()
    for task in by_arm["demo_1"]:
        assert task.point.loss_type == "demo_1"
        assert task.point.normalize_agent_reward is True
        assert task.point.pref_budget == 0
        assert task.demo_subsample_seed == 1000 + task.seed
    for task in by_arm["demo_2_no_norm"]:
        assert task.point.loss_type == "demo_2"
        assert task.point.normalize_agent_reward is False
        assert task.point.pref_budget == 0
        assert task.demo_subsample_seed == 1000 + task.seed


def test_task_overrides_win_with_exact_runtime_and_wandb_metadata():
    tasks = orchestrator.build_tasks()
    representative = {
        task.arm: task
        for task in reversed(tasks)
    }
    for arm, task in representative.items():
        values = overrides_dict(task)
        assert values["run.name"] == task.run_name
        assert values["run.group"] == task.group
        assert values["run.seed"] == str(task.seed)
        assert values["wandb.project"] == orchestrator.WANDB_PROJECT
        assert values["algo.kwargs.total_queries"] == str(
            task.point.pref_budget
        )
        assert values["algo.kwargs.initial_queries"] == str(
            task.point.initial_queries
        )
        assert values["train.kwargs.total_timesteps"] == "2000000"
        assert values["train.kwargs.timesteps_per_iteration"] == "20000"
        assert values["env.n_envs"] == "2"
        if task.point.demo_budget is not None:
            assert values["run.n_expert_trajectories"] == str(
                task.point.demo_budget
            )
            assert values["run.demo_subsample_seed"] == str(1000 + task.seed)
        else:
            assert "run.demo_subsample_seed" not in values


def test_training_command_and_environment_are_pinned_and_resumable():
    task = orchestrator.build_tasks()[0]
    command = orchestrator.build_training_command(
        task,
        "24",
        base_overrides=("algo.kwargs.lr_rew=0.1",),
    )
    environment = orchestrator.training_environment(task)

    assert command[:5] == [
        "taskset",
        "-c",
        "24",
        str(orchestrator.PYTHON),
        "scripts/train_hybrid_sac.py",
    ]
    assert command[5] == "algo.kwargs.lr_rew=0.1"
    assert command[-len(orchestrator.task_overrides(task)) :] == list(
        orchestrator.task_overrides(task)
    )
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["MKL_NUM_THREADS"] == "1"
    assert environment["OPENBLAS_NUM_THREADS"] == "1"
    assert environment["WANDB_MODE"] == "online"
    assert environment["WANDB_RUN_ID"] == task.wandb_run_id
    assert environment["WANDB_RESUME"] == "allow"


def test_training_environment_overrides_offline_and_disabled(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DISABLED", "true")

    environment = orchestrator.training_environment(
        orchestrator.build_tasks()[0]
    )

    assert environment["WANDB_MODE"] == "online"
    assert "WANDB_DISABLED" not in environment


def test_final_output_validation_rejects_partial_or_invalid_files(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "final_eval.json").write_text("{broken")
    (directory / "agent_final.zip").write_bytes(b"not a zip")
    assert not orchestrator.valid_final_output(directory)

    (directory / "final_eval.json").write_text(
        json.dumps(
            {
                "eval/mean_fast_return": 1.0,
                "eval/success_rate": float("nan"),
            }
        )
    )
    with zipfile.ZipFile(directory / "agent_final.zip", "w") as archive:
        archive.writestr("policy.pth", b"test")
    assert not orchestrator.valid_final_output(directory)


def test_reap_waits_for_process_exit_before_marking_done(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(orchestrator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(orchestrator, "RUN_ROOT", tmp_path / "runs")
    monkeypatch.setattr(orchestrator, "LOG_ROOT", tmp_path / "logs")
    monkeypatch.setattr(orchestrator, "MARKER_ROOT", tmp_path / "markers")
    task = orchestrator.build_tasks()[0]
    write_valid_output(task)
    orchestrator.write_marker(task, "running", attempts=1)
    legacy = fake_process(task)
    state = transition_state(
        (task,), (legacy,), phase=orchestrator.PHASE_SINGLE
    )
    scheduler = orchestrator.Scheduler((task,), 1, 2, 1, state)
    scheduler.running[task.key] = orchestrator.Running(
        task, 1234, "24", FakeProcess([None, 0])
    )

    scheduler.reap([])
    assert task.key in scheduler.running
    assert not orchestrator.task_done(task)

    scheduler.reap([])
    assert task.key not in scheduler.running
    assert orchestrator.task_done(task)
    assert orchestrator.read_marker(task)["return_code"] == 0


def test_reap_retries_nonzero_exit_even_with_valid_artifacts(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(orchestrator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(orchestrator, "RUN_ROOT", tmp_path / "runs")
    monkeypatch.setattr(orchestrator, "LOG_ROOT", tmp_path / "logs")
    monkeypatch.setattr(orchestrator, "MARKER_ROOT", tmp_path / "markers")
    task = orchestrator.build_tasks()[0]
    write_valid_output(task)
    orchestrator.write_marker(task, "running", attempts=1)
    legacy = fake_process(task)
    state = transition_state(
        (task,), (legacy,), phase=orchestrator.PHASE_SINGLE
    )
    scheduler = orchestrator.Scheduler((task,), 1, 2, 1, state)
    scheduler.running[task.key] = orchestrator.Running(
        task, 1234, "24", FakeProcess([1])
    )

    scheduler.reap([])

    assert not orchestrator.task_done(task)
    assert orchestrator.read_marker(task)["state"] == "retry_pending"


def test_slots_partition_cores_24_through_47():
    legacy_cores = [
        core
        for slot in orchestrator.LEGACY_PAIR_SLOTS
        for core in orchestrator.slot_cores(slot)
    ]
    single_cores = [
        core
        for slot in orchestrator.SINGLE_SLOTS
        for core in orchestrator.slot_cores(slot)
    ]
    assert len(orchestrator.LEGACY_PAIR_SLOTS) == 12
    assert len(orchestrator.SINGLE_SLOTS) == 24
    assert legacy_cores == list(range(24, 48))
    assert single_cores == list(range(24, 48))
    assert orchestrator.slot_cores("24") == frozenset({24})
    assert orchestrator.slot_cores("24-25") == frozenset({24, 25})


def test_hybrid_points_are_first_and_three_seeds_stay_adjacent():
    tasks = orchestrator.build_tasks()
    assert all(task.arm.startswith("hybrid_") for task in tasks[:27])
    assert all(not task.arm.startswith("hybrid_") for task in tasks[27:])

    for offset in range(0, len(tasks), 3):
        triplet = tasks[offset : offset + 3]
        assert len({task.group for task in triplet}) == 1
        assert [task.seed for task in triplet] == [1, 2, 3]


def test_approved_optuna_sources_are_frozen():
    expected = {
        "demo_1": ("hybrid_sac_demo_1", 24),
        "demo_2_no_norm": ("hybrid_sac_demo_2_no_norm", 26),
        "pref_soft": ("hybrid_sac_pref_soft", 3),
        "pref_bernoulli": (
            "hybrid_sac_pref_bernoulli_q100k_temp",
            21,
        ),
        "hybrid_demo_2_soft": ("hybrid_sac_hybrid_demo_2", 10),
        "hybrid_demo_2_bernoulli": (
            "hybrid_sac_hybrid_demo_2_bernoulli_norm",
            11,
        ),
    }
    assert {
        key: (source.study_name, source.expected_trial)
        for key, source in orchestrator.CONFIG_SOURCES.items()
    } == expected


def test_singleton_commands_cover_first_24_tasks_once(monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "export_overrides",
        lambda *_args: ("algo.kwargs.lr_rew=0.1",),
    )
    records = orchestrator.command_records(orchestrator.build_tasks())

    assert [record["planned_slot"] for record in records[:24]] == list(
        orchestrator.SINGLE_SLOTS
    )
    assert all(
        f"taskset -c {record['planned_slot']}" in record["command"]
        for record in records
    )
    with pytest.raises(ValueError, match="Unknown CPU slot"):
        orchestrator.build_training_command(
            orchestrator.build_tasks()[0],
            "24-25",
            base_overrides=("algo.kwargs.lr_rew=0.1",),
        )


def test_capture_transition_preserves_process_identity(
    isolated_state, monkeypatch
):
    task = orchestrator.build_tasks()[0]
    trainer = fake_process(task)
    manager = orchestrator.ProcessInfo(
        pid=999,
        ppid=1,
        pgid=999,
        sid=999,
        start_time=111,
        state="S",
        affinity=frozenset(range(48)),
        command=(
            "/python scripts/schedule_budget_curves_completion.py"
        ),
    )
    monkeypatch.setattr(
        orchestrator.os,
        "sched_setaffinity",
        lambda *_args: pytest.fail("capture must not repin a trainer"),
        raising=False,
    )

    state = orchestrator.capture_transition_state(
        (task,), [manager, trainer], manager.pid, expected_live=1
    )

    member = state["cohort"][0]
    assert state["phase"] == orchestrator.PHASE_DRAIN
    assert member["pid"] == trainer.pid
    assert member["start_time"] == trainer.start_time
    assert member["pgid"] == trainer.pgid
    assert member["sid"] == trainer.sid
    assert member["affinity"] == [24, 25]
    assert (
        orchestrator.load_transition_state((task,))["cohort"][0]
        == member
    )


def test_duplicate_live_run_names_fail_closed():
    task = orchestrator.build_tasks()[0]
    first = fake_process(task, pid=100)
    second = fake_process(
        task, pid=101, affinity=(26, 27), start_time=457
    )

    with pytest.raises(RuntimeError, match="Duplicate live run.name"):
        orchestrator.live_training_runs([first, second])


def test_hydra_config_resolution_is_not_a_live_training_run():
    task = orchestrator.build_tasks()[0]
    config_only = fake_process(
        task,
        affinity=tuple(range(48)),
        command=(
            "/python scripts/train_hybrid_sac.py --cfg job --resolve "
            f"run.name={task.run_name}"
        ),
    )

    assert orchestrator.live_training_runs([config_only]) == {}


def test_capture_can_recover_from_verified_stopped_legacy_manager(
    isolated_state
):
    task = orchestrator.build_tasks()[0]
    trainer = fake_process(task)
    orchestrator.LOCK_PATH.parent.mkdir(parents=True)
    orchestrator.LOCK_PATH.write_text("999\n")

    state = orchestrator.capture_transition_state(
        (task,),
        [trainer],
        legacy_manager_pid=999,
        expected_live=1,
        allow_stopped_manager=True,
    )

    assert state["legacy_manager"]["pid"] == 999
    assert (
        state["legacy_manager"]["state"]
        == "already_stopped_before_capture"
    )
    assert state["legacy_manager"]["stale_lock_verified"] is True


def test_drain_never_launches_while_legacy_group_is_alive(
    isolated_state, monkeypatch
):
    task = orchestrator.build_tasks()[0]
    trainer = fake_process(task)
    state = transition_state((task,), (trainer,))
    scheduler = orchestrator.Scheduler((task,), 24, 2, 1, state)
    monkeypatch.setattr(orchestrator, "iter_processes", lambda: [trainer])
    monkeypatch.setattr(
        scheduler,
        "launch",
        lambda *_args: pytest.fail("drain must never launch"),
    )

    switched = scheduler.drain_step([trainer])

    assert switched is False
    assert scheduler.phase == orchestrator.PHASE_DRAIN
    assert task.key in scheduler.running
    assert orchestrator.attempts(task) == 1


def test_descendant_keeps_barrier_until_entire_process_group_exits(
    isolated_state, monkeypatch
):
    task = orchestrator.build_tasks()[0]
    trainer = fake_process(task)
    descendant = orchestrator.ProcessInfo(
        pid=2000,
        ppid=trainer.pid,
        pgid=trainer.pgid,
        sid=trainer.sid,
        start_time=999,
        state="S",
        affinity=trainer.affinity,
        command="/python -c multiprocessing.worker",
    )
    state = transition_state((task,), (trainer,))
    scheduler = orchestrator.Scheduler((task,), 24, 2, 1, state)
    monkeypatch.setattr(orchestrator, "iter_processes", lambda: [descendant])

    assert scheduler.drain_step([descendant]) is False
    assert scheduler.phase == orchestrator.PHASE_DRAIN

    monkeypatch.setattr(orchestrator, "iter_processes", lambda: [])
    assert scheduler.drain_step([]) is True
    assert scheduler.phase == orchestrator.PHASE_SINGLE
    assert orchestrator.read_marker(task)["state"] == "retry_pending"


def test_wandb_helpers_in_separate_sessions_belong_to_legacy_run(
    isolated_state, monkeypatch
):
    task = orchestrator.build_tasks()[0]
    trainer = fake_process(task)
    wandb_core = orchestrator.ProcessInfo(
        pid=2001,
        ppid=1,
        pgid=2001,
        sid=2001,
        start_time=777,
        state="S",
        affinity=trainer.affinity,
        command=(
            "/wandb/bin/wandb-core --port-filename /tmp/port "
            f"--pid {trainer.pid}"
        ),
    )
    gpu_stats = orchestrator.ProcessInfo(
        pid=2002,
        ppid=wandb_core.pid,
        pgid=2002,
        sid=2002,
        start_time=778,
        state="S",
        affinity=trainer.affinity,
        command=(
            "/wandb/bin/gpu_stats --portfile /tmp/gpu "
            f"--parent-pid {wandb_core.pid}"
        ),
    )
    state = transition_state((task,), (trainer,))
    scheduler = orchestrator.Scheduler((task,), 24, 2, 1, state)
    helpers = [wandb_core, gpu_stats]
    monkeypatch.setattr(orchestrator, "iter_processes", lambda: helpers)

    owned = orchestrator.owned_run_processes(
        [trainer, *helpers],
        root_pid=trainer.pid,
        pgid=trainer.pgid,
        sid=trainer.sid,
        slot="24-25",
    )
    assert {process.pid for process in owned} == {
        trainer.pid,
        wandb_core.pid,
        gpu_stats.pid,
    }

    # Even after the trainer/group disappears, the explicit W&B --pid link
    # keeps the drain barrier closed until both helper sessions exit.
    assert scheduler.drain_step(helpers) is False
    assert scheduler.phase == orchestrator.PHASE_DRAIN

    monkeypatch.setattr(orchestrator, "iter_processes", lambda: [gpu_stats])
    assert scheduler.drain_step([gpu_stats]) is False
    assert scheduler.phase == orchestrator.PHASE_DRAIN

    monkeypatch.setattr(orchestrator, "iter_processes", lambda: [])
    assert scheduler.drain_step([]) is True
    assert scheduler.phase == orchestrator.PHASE_SINGLE


def test_completion_in_manager_gap_becomes_done_without_relaunch(
    isolated_state, monkeypatch
):
    task = orchestrator.build_tasks()[0]
    trainer = fake_process(task)
    state = transition_state((task,), (trainer,))
    orchestrator.write_marker(task, "running", attempts=1)
    write_valid_output(task)
    scheduler = orchestrator.Scheduler((task,), 24, 2, 1, state)
    monkeypatch.setattr(orchestrator, "iter_processes", lambda: [])
    monkeypatch.setattr(
        orchestrator.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("must not relaunch"),
    )

    assert scheduler.drain_step([]) is True

    assert scheduler.phase == orchestrator.PHASE_SINGLE
    assert orchestrator.task_done(task)
    assert scheduler.pending(set()) == []


def test_failed_legacy_is_held_for_singleton_retry_until_barrier(
    isolated_state, monkeypatch
):
    first, second = orchestrator.build_tasks()[:2]
    first_process = fake_process(first, pid=100, affinity=(24, 25))
    second_process = fake_process(
        second, pid=200, affinity=(26, 27), start_time=789
    )
    state = transition_state(
        (first, second), (first_process, second_process)
    )
    orchestrator.write_marker(first, "running", attempts=1)
    orchestrator.write_marker(second, "running", attempts=1)
    scheduler = orchestrator.Scheduler(
        (first, second), 24, 2, 1, state
    )
    monkeypatch.setattr(
        orchestrator, "iter_processes", lambda: [second_process]
    )

    assert scheduler.drain_step([second_process]) is False

    assert orchestrator.read_marker(first)["state"] == "retry_pending"
    assert scheduler.pending({second.run_name}) == [first]
    assert scheduler.phase == orchestrator.PHASE_DRAIN


def test_pid_reuse_is_rejected_during_legacy_adoption(isolated_state):
    task = orchestrator.build_tasks()[0]
    original = fake_process(task, start_time=456)
    reused = fake_process(task, start_time=999)
    state = transition_state((task,), (original,))
    scheduler = orchestrator.Scheduler((task,), 24, 2, 1, state)

    with pytest.raises(RuntimeError, match="identity mismatch"):
        scheduler.adopt_legacy([reused])


def test_pid_reuse_is_rejected_for_singleton_marker(isolated_state):
    task = orchestrator.build_tasks()[0]
    legacy = fake_process(task)
    state = transition_state(
        (task,), (legacy,), phase=orchestrator.PHASE_SINGLE
    )
    orchestrator.write_marker(
        task,
        "running",
        attempts=1,
        pid=1234,
        start_time=456,
        pgid=1234,
        sid=1234,
        slot="24",
    )
    reused = fake_process(task, affinity=(24,), start_time=999)
    scheduler = orchestrator.Scheduler((task,), 24, 2, 1, state)

    with pytest.raises(RuntimeError, match="marker identity mismatch"):
        scheduler.validate_live_layout([reused])


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda state: state.update(version=999), "version"),
        (
            lambda state: state.update(matrix_sha256="wrong"),
            "matrix",
        ),
        (lambda state: state.update(phase="back_to_pairs"), "phase"),
    ],
)
def test_corrupt_or_incompatible_transition_state_fails_closed(
    isolated_state, mutation, error
):
    task = orchestrator.build_tasks()[0]
    state = transition_state((task,), (fake_process(task),))
    mutation(state)
    orchestrator.atomic_json(orchestrator.TRANSITION_STATE_PATH, state)

    with pytest.raises(RuntimeError, match=error):
        orchestrator.load_transition_state((task,))


def test_budget_replacement_migrates_legacy_transition_once(
    isolated_state, monkeypatch
):
    legacy_tasks = orchestrator.build_tasks(
        pref_bernoulli_points=orchestrator.LEGACY_PREF_BERNOULLI_POINTS
    )
    tasks = orchestrator.build_tasks()
    legacy = fake_process(legacy_tasks[0])
    state = transition_state(
        legacy_tasks, (legacy,), phase=orchestrator.PHASE_SINGLE
    )
    orchestrator.atomic_json(orchestrator.TRANSITION_STATE_PATH, state)
    monkeypatch.setattr(orchestrator, "iter_processes", lambda: [])

    result = orchestrator.migrate_pref_bernoulli_budget(tasks)
    migrated = orchestrator.load_transition_state(tasks)

    assert result["matrix_sha256"] == orchestrator.task_matrix_sha256(tasks)
    assert result == migrated
    assert migrated["history"][-1]["event"] == (
        "migrated_pref_bernoulli_budget_15000_to_1000000"
    )

    # Repeating the dedicated migration command is safe and does not append
    # another migration event.
    repeated = orchestrator.migrate_pref_bernoulli_budget(tasks)
    assert repeated == migrated
    assert orchestrator.load_transition_state(tasks) == migrated


def test_budget_replacement_migration_rejects_started_legacy_task(
    isolated_state, monkeypatch
):
    legacy_tasks = orchestrator.build_tasks(
        pref_bernoulli_points=orchestrator.LEGACY_PREF_BERNOULLI_POINTS
    )
    tasks = orchestrator.build_tasks()
    legacy = fake_process(legacy_tasks[0])
    state = transition_state(
        legacy_tasks, (legacy,), phase=orchestrator.PHASE_SINGLE
    )
    orchestrator.atomic_json(orchestrator.TRANSITION_STATE_PATH, state)
    old_high_budget = next(
        task
        for task in legacy_tasks
        if task.arm == "pref_bernoulli" and task.budget == 15_000
    )
    orchestrator.write_marker(
        old_high_budget,
        "retry_pending",
        attempts=1,
    )
    before = orchestrator.TRANSITION_STATE_PATH.read_text()
    monkeypatch.setattr(orchestrator, "iter_processes", lambda: [])

    with pytest.raises(RuntimeError, match="15.*000|legacy|started"):
        orchestrator.migrate_pref_bernoulli_budget(tasks)

    assert orchestrator.TRANSITION_STATE_PATH.read_text() == before


def test_budget_replacement_migration_rejects_unknown_matrix_hash(
    isolated_state, monkeypatch
):
    legacy_tasks = orchestrator.build_tasks(
        pref_bernoulli_points=orchestrator.LEGACY_PREF_BERNOULLI_POINTS
    )
    tasks = orchestrator.build_tasks()
    legacy = fake_process(legacy_tasks[0])
    state = transition_state(
        legacy_tasks, (legacy,), phase=orchestrator.PHASE_SINGLE
    )
    state["matrix_sha256"] = "unexpected"
    orchestrator.atomic_json(orchestrator.TRANSITION_STATE_PATH, state)
    before = orchestrator.TRANSITION_STATE_PATH.read_text()
    monkeypatch.setattr(orchestrator, "iter_processes", lambda: [])

    with pytest.raises(RuntimeError, match="matrix|hash"):
        orchestrator.migrate_pref_bernoulli_budget(tasks)

    assert orchestrator.TRANSITION_STATE_PATH.read_text() == before


def test_max_hybrid_point_migrates_previous_matrix(
    isolated_state, monkeypatch
):
    previous_tasks = orchestrator.build_tasks(
        include_max_hybrid_bernoulli=False
    )
    tasks = orchestrator.build_tasks()
    legacy = fake_process(previous_tasks[0])
    state = transition_state(
        previous_tasks, (legacy,), phase=orchestrator.PHASE_SINGLE
    )
    orchestrator.atomic_json(orchestrator.TRANSITION_STATE_PATH, state)
    monkeypatch.setattr(orchestrator, "iter_processes", lambda: [])

    migrated = orchestrator.migrate_hybrid_bernoulli_max_budget(tasks)

    assert migrated["matrix_sha256"] == orchestrator.task_matrix_sha256(tasks)
    assert migrated["history"][-1]["event"] == (
        "added_hybrid_demo_2_bernoulli_pref2723_demo2723"
    )
    assert orchestrator.load_transition_state(tasks) == migrated


def test_same_filename_orchestrator_is_not_silently_ignored():
    task = orchestrator.build_tasks()[0]
    process = fake_process(
        task,
        pid=999,
        command=(
            "/python scripts/schedule_budget_curves_completion.py "
            "--max-parallel 24"
        ),
    )

    assert orchestrator.other_orchestrators([process]) == [process]


def test_parent_shell_mentioning_orchestrator_is_not_a_second_manager():
    task = orchestrator.build_tasks()[0]
    shell = fake_process(
        task,
        pid=998,
        command=(
            "bash -c /python "
            "scripts/schedule_budget_curves_completion.py.next --preflight"
        ),
    )

    assert orchestrator.other_orchestrators([shell]) == []


@pytest.mark.parametrize("mode", ("--status", "--preflight", "--dry-run"))
def test_read_only_orchestrator_process_is_not_a_competing_manager(mode):
    task = orchestrator.build_tasks()[0]
    process = fake_process(
        task,
        pid=997,
        command=(
            "/python scripts/schedule_budget_curves_completion.py "
            f"{mode}"
        ),
    )

    assert orchestrator.other_orchestrators([process]) == []


def test_single_phase_reconciles_stale_running_marker(
    isolated_state
):
    task = orchestrator.build_tasks()[0]
    legacy = fake_process(task)
    state = transition_state(
        (task,), (legacy,), phase=orchestrator.PHASE_SINGLE
    )
    orchestrator.write_marker(
        task,
        "running",
        attempts=1,
        pid=4321,
        start_time=987,
        pgid=4321,
        sid=4321,
        slot="24",
    )
    write_valid_output(task)
    scheduler = orchestrator.Scheduler((task,), 24, 2, 1, state)

    scheduler.reconcile_stale_markers(set(), [])

    assert orchestrator.task_done(task)


def test_single_phase_adopts_detached_run_despite_stale_marker(
    isolated_state
):
    task = orchestrator.build_tasks()[0]
    legacy = fake_process(task)
    state = transition_state(
        (task,), (legacy,), phase=orchestrator.PHASE_SINGLE
    )
    orchestrator.write_marker(
        task,
        "running",
        attempts=1,
        pid=1234,
        start_time=456,
        pgid=1234,
        sid=1234,
        slot="24",
    )
    detached = fake_process(
        task,
        pid=5678,
        affinity=(24,),
        start_time=789,
        pgid=5678,
        sid=5678,
    )
    scheduler = orchestrator.Scheduler((task,), 24, 2, 1, state)

    scheduler.validate_live_layout([detached])
    scheduler.adopt_singletons([detached])

    running = scheduler.running[task.key]
    marker = orchestrator.read_marker(task)
    assert running.adopted is True
    assert running.pid == detached.pid
    assert running.slot == "24"
    assert marker["state"] == "running"
    assert marker["pid"] == detached.pid
    assert marker["start_time"] == detached.start_time
    assert scheduler.pending({task.run_name}) == []


def test_external_process_reserves_only_its_core_without_crashing(
    isolated_state, monkeypatch
):
    tasks = orchestrator.build_tasks()[:24]
    legacy = fake_process(tasks[0])
    state = transition_state(
        tasks, (legacy,), phase=orchestrator.PHASE_SINGLE
    )
    state["rollout"] = "full"
    scheduler = orchestrator.Scheduler(tasks, 24, 2, 1, state)
    scheduler.idle_scans = {
        slot: orchestrator.IDLE_SCANS_REQUIRED
        for slot in orchestrator.SINGLE_SLOTS
    }
    external = orchestrator.ProcessInfo(
        pid=9000,
        ppid=1,
        pgid=9000,
        sid=9000,
        start_time=123,
        state="R",
        affinity=frozenset({24}),
        command="/python external_ppo_job.py",
    )
    launches = []

    def fake_launch(task, slot, _processes):
        launches.append((task, slot))
        scheduler.running[task.key] = orchestrator.Running(
            task=task,
            pid=10_000 + len(launches),
            slot=slot,
        )
        return True

    monkeypatch.setattr(
        orchestrator, "iter_processes", lambda: [external]
    )
    monkeypatch.setattr(
        orchestrator,
        "slot_cpu_busy",
        lambda: {
            slot: (0.0, 0.0) for slot in orchestrator.SINGLE_SLOTS
        },
    )
    monkeypatch.setattr(scheduler, "launch", fake_launch)

    assert scheduler.single_step([external]) is False

    launched_slots = [slot for _task, slot in launches]
    assert "24" not in launched_slots
    assert launched_slots == list(orchestrator.SINGLE_SLOTS[1:])
    assert len({task.key for task, _slot in launches}) == len(launches)
    assert len(set(launched_slots)) == len(launched_slots)


@pytest.mark.parametrize(
    ("rollout", "expected_launches"),
    [("canary", 1), ("full", 24)],
)
def test_single_phase_respects_canary_and_24_run_limit(
    isolated_state, monkeypatch, rollout, expected_launches
):
    tasks = orchestrator.build_tasks()[:25]
    legacy = fake_process(tasks[0])
    state = transition_state(
        tasks, (legacy,), phase=orchestrator.PHASE_SINGLE
    )
    state["rollout"] = rollout
    scheduler = orchestrator.Scheduler(tasks, 24, 2, 1, state)
    scheduler.idle_scans = {
        slot: orchestrator.IDLE_SCANS_REQUIRED
        for slot in orchestrator.SINGLE_SLOTS
    }
    launches = []

    def fake_launch(task, slot, _processes):
        launches.append((task, slot))
        scheduler.running[task.key] = orchestrator.Running(
            task=task,
            pid=1000 + len(launches),
            slot=slot,
        )
        return True

    monkeypatch.setattr(orchestrator, "iter_processes", lambda: [])
    monkeypatch.setattr(
        orchestrator,
        "slot_cpu_busy",
        lambda: {
            slot: (0.0, 0.0) for slot in orchestrator.SINGLE_SLOTS
        },
    )
    monkeypatch.setattr(scheduler, "launch", fake_launch)

    assert scheduler.single_step([]) is False

    assert len(launches) == expected_launches
    assert [slot for _task, slot in launches] == list(
        orchestrator.SINGLE_SLOTS[:expected_launches]
    )


def test_canary_requires_wandb_and_first_iteration_before_ramp(
    isolated_state
):
    task = orchestrator.build_tasks()[0]
    legacy = fake_process(task)
    state = transition_state(
        (task,), (legacy,), phase=orchestrator.PHASE_SINGLE
    )
    state.update(
        {
            "rollout": "canary",
            "canary_task_key": task.key,
            "canary_pid": 4321,
            "canary_started_at_epoch": time.time()
            - orchestrator.CANARY_SECONDS
            - 1,
            "canary_log_offset": 0,
        }
    )
    scheduler = orchestrator.Scheduler((task,), 24, 2, 1, state)
    scheduler.running[task.key] = orchestrator.Running(
        task=task,
        pid=4321,
        slot="24",
    )
    task.log_path.parent.mkdir(parents=True)
    task.log_path.write_text(
        "wandb: Tracking run with wandb version 0.25.1\n"
    )

    scheduler.maybe_promote_canary()
    assert scheduler.transition_state["rollout"] == "canary"

    with task.log_path.open("a") as stream:
        stream.write("Iteration 0/99\n")
    scheduler.maybe_promote_canary()

    assert scheduler.transition_state["rollout"] == "full"


def test_stop_handler_changes_only_scheduler_flag(monkeypatch):
    monkeypatch.setattr(orchestrator, "STOP_REQUESTED", False)
    monkeypatch.setattr(
        orchestrator.os,
        "kill",
        lambda *_args: pytest.fail("handler must not signal trainers"),
    )

    orchestrator.handle_stop(15, None)

    assert orchestrator.STOP_REQUESTED is True
