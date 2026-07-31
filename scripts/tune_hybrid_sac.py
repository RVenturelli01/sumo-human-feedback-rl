"""Optuna hyperparameter search for the HybridAlgorithm SAC arms.

Six arms, one Optuna study each (sharing one journal file):

* ``pref_soft``      — preferences only, soft oracle labels
* ``pref_bernoulli`` — preferences only, sampled binary oracle labels
* ``demo_1``         — demonstrations only, difference-of-means loss
* ``demo_2``         — demonstrations only, MaxEnt-surrogate loss
* ``hybrid_demo_1``  — preferences + demonstrations, demo loss demo_1
* ``hybrid_demo_2``  — preferences + demonstrations, demo loss demo_2

Hybrid arms use soft labels by default. Pass
``--preference-labels binary_bernoulli`` with a dedicated ``--study-suffix``
to tune the Bernoulli variant, including ``pref_temperature`` and the
bootstrap size ``initial_queries``.

Each trial runs ``scripts/train_hybrid_sac.py`` as a subprocess (libsumo, W&B
and SubprocVecEnv state are all cleaned up by process exit), follows the
per-iteration ``metrics.jsonl`` written by the training run for median
pruning, and reads the final held-out ``final_eval.json`` as the objective
(``eval/mean_fast_return``, maximized).

Recommended setup: ONE worker per arm (see run_optuna_parallel_arms.sh) so
every study is fully sequential — after the random startup trials, each new
trial is informed by all completed ones. Multiple workers on the same arm are
also supported (shared journal). Every override is passed explicitly so
trials never depend on yaml/launcher default drift.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_hybrid_sac.py"

ARMS = ("pref_soft", "pref_bernoulli", "demo_1", "demo_2", "hybrid_demo_1", "hybrid_demo_2")
PREFERENCE_LABEL_CHOICES = ("auto", "soft", "binary_bernoulli")
# --- EXTENSION PLACEHOLDER: literature hybrid baseline (Ibarz 2018) ---------
# "Demonstrations as implicit preferences" is already implemented and tested
# in HybridAlgorithm as demo_mode="preferences" (expert>agent pairs mixed into
# one Bradley-Terry objective). To compare against it, add an "ibarz" arm:
#   * ARMS += ("ibarz",)
#   * arm_overrides: like the hybrid arms but with
#       algo.kwargs.demo_mode=preferences (instead of gcl),
#       algo.kwargs.demo_pref_pairs_per_iteration=<budget-derived>,
#       algo.kwargs.demo_pref_batch_fraction=0.5 (or tuned),
#     and no demo_weight (unused in that mode);
#   * suggest_params: pref-side params + demo_pref_batch_fraction.
# Everything downstream (pruning, export, final runs, reports) works as-is.
PRUNE_METRIC = "rollout/mean_true_reward"
OBJECTIVE_METRIC = "eval/mean_fast_return"
POLL_SECONDS = 30
KILL_GRACE_SECONDS = 30

# Collapse rule. A hybrid run whose policy crashes immediately produces
# ~1-step episodes, so SUMO resets dominate and the trial takes ~24h instead
# of ~4h while pinned at the reward floor. MedianPruner cannot catch this in
# the first wave (with N concurrent workers the first N trials all start
# before any completes), so this is a separate, absolute rule.
COLLAPSE_REWARD = -45.0
COLLAPSE_MIN_STEP = 10
COLLAPSE_STREAK = 5

# Search-space identity. Bump when suggest_params changes so a study cannot
# silently mix trials drawn from different spaces.
SEARCH_SPACE_VERSION = 2

# Values fixed across all arms and trials. pref_temperature and
# preference_fragment_length define the synthetic oracle, i.e. the problem,
# not the learner. SAC kwargs come from the true-reward baseline tuning;
# gradient_steps=32 preserves the historical replay ratio of 2.0 at n_envs=2
# (train_freq=8 -> 16 transitions per cycle).
FIXED_OVERRIDES = [
    "env.kwargs.ego=continuous",
    "env.kwargs.reward=fast",
    "agent.kwargs.learning_rate=0.0001242983309370202",
    "agent.kwargs.buffer_size=300000",
    "agent.kwargs.learning_starts=2000",
    "agent.kwargs.batch_size=256",
    "agent.kwargs.gamma=0.995",
    "agent.kwargs.tau=0.005",
    "agent.kwargs.ent_coef=auto",
    "agent.kwargs.train_freq=8",
    "agent.kwargs.gradient_steps=32",
    "agent.kwargs.policy_kwargs.net_arch=[64,64]",
    "agent.kwargs.device=cpu",
    "algo.kwargs.demo_mode=gcl",
    "algo.kwargs.relabel_rewards=true",
    "algo.kwargs.normalize_agent_reward=true",
    "algo.kwargs.pref_temperature=20.0",
    "algo.kwargs.preference_fragment_length=1",
    "algo.kwargs.train_comparison_frac=0.8",
    "algo.kwargs.exploration_frac=0.0",
    "algo.kwargs.agent_log_timestep_interval=10000",
    "algo.kwargs.reward_model_kwargs.n_ensembles=3",
    "algo.kwargs.reward_model_kwargs.activation_fn=tanh",
    "algo.kwargs.reward_model_kwargs.alpha=1",
    "train.kwargs.log_interval=100",
    "train.kwargs.checkpoint_interval=1000000",
    "train.kwargs.scatter_interval=0",
]

NET_ARCH_CHOICES = ["[8,8]", "[16,16]", "[32,32]", "[64,64]", "[128,128]"]


def uses_preferences(arm: str) -> bool:
    return arm.startswith("pref_") or arm.startswith("hybrid_")


def uses_demos(arm: str) -> bool:
    return arm.startswith("demo_") or arm.startswith("hybrid_")


def resolve_preference_labels(arm: str, preference_labels: str = "auto") -> str:
    """Resolve ``auto`` while preserving the historical per-arm defaults."""
    if preference_labels not in PREFERENCE_LABEL_CHOICES:
        raise ValueError(
            f"Unsupported preference labels {preference_labels!r}; "
            f"choose one of {PREFERENCE_LABEL_CHOICES}."
        )
    if preference_labels != "auto":
        return preference_labels
    return "binary_bernoulli" if arm == "pref_bernoulli" else "soft"


def arm_overrides(
    arm: str,
    pref_budget: int,
    demo_budget: int,
    preference_labels: str = "auto",
) -> list:
    """Arm-defining overrides (the launcher MODE presets, made explicit)."""
    overrides = []
    if uses_preferences(arm):
        labels = resolve_preference_labels(arm, preference_labels)
        overrides += [
            f"algo.kwargs.labels_type={labels}",
            f"algo.kwargs.total_queries={pref_budget}",
            f"train.kwargs.total_queries={pref_budget}",
        ]
    else:
        overrides += [
            "algo.kwargs.labels_type=soft",  # inert with zero queries
            "algo.kwargs.total_queries=0",
            "algo.kwargs.initial_queries=0",
            "train.kwargs.total_queries=0",
            "algo.kwargs.query_schedule=constant",
            "algo.kwargs.fragmenter_type=active",
            "algo.kwargs.batch_size_pref=128",  # inert with zero queries
        ]
    if uses_demos(arm):
        loss = "demo_1" if arm.endswith("demo_1") else "demo_2"
        overrides += [
            f"algo.kwargs.loss_type={loss}",
            f"run.n_expert_trajectories={demo_budget}",
        ]
        if arm.startswith("demo_"):
            overrides.append("algo.kwargs.demo_weight=1.0")
    else:
        overrides += [
            "algo.kwargs.demo_weight=0.0",
            "algo.kwargs.loss_type=demo_2",  # inert with demo_weight=0
            "algo.kwargs.batch_size_expert=64",  # inert with demo_weight=0
            "algo.kwargs.batch_size_model=64",
        ]
    if arm.startswith("hybrid_"):
        # Kept small for the hybrid arms so the search stays tractable
        # (~8 sampled params); revisit with --override if needed.
        overrides += [
            "algo.kwargs.query_schedule=constant",
            f"algo.kwargs.initial_queries={max(100, round(0.1 * pref_budget))}",
            "algo.kwargs.fragmenter_type=active",
            "algo.kwargs.batch_size_model=64",
        ]
    return overrides


def initial_queries_choices(pref_budget: int) -> list:
    """Bootstrap-chunk choices as 2-20% of the budget (=[100,250,500,1000] at 5k)."""
    return sorted({max(100, round(pref_budget * f)) for f in (0.02, 0.05, 0.1, 0.2)})


def fixed_param_overrides(
    fix_demo_weight=None,
    fix_pref_temperature=None,
) -> list:
    """Hydra overrides for params pinned on the CLI and dropped from the search.

    Emitted AFTER params_to_overrides so they beat FIXED_OVERRIDES (which pins
    pref_temperature=20.0) and arm_overrides; --override still wins last.
    """
    overrides = []
    if fix_demo_weight is not None:
        overrides.append(f"algo.kwargs.demo_weight={fix_demo_weight!r}")
    if fix_pref_temperature is not None:
        overrides.append(f"algo.kwargs.pref_temperature={fix_pref_temperature!r}")
    return overrides


def suggest_params(
    trial: optuna.Trial,
    arm: str,
    pref_budget: int = 5000,
    preference_labels: str = "auto",
    fix_demo_weight=None,
    fix_pref_temperature=None,
) -> dict:
    """Sample the per-arm search space; returns {param_name: value}.

    Params pinned via fix_* are NOT sampled: leaving them in the space would
    waste search dimensions on values that the overrides then ignore.
    """
    resolved_labels = (
        resolve_preference_labels(arm, preference_labels)
        if uses_preferences(arm)
        else None
    )
    params = {
        "lr_rew": trial.suggest_float("lr_rew", 3e-5, 3e-3, log=True),
        "gradient_steps_rew": trial.suggest_int("gradient_steps_rew", 20, 400, log=True),
        "l2_rew": trial.suggest_float("l2_rew", 1e-6, 1e-3, log=True),
        "reward_net_arch": trial.suggest_categorical("reward_net_arch", NET_ARCH_CHOICES),
        "initial_agent_timesteps": trial.suggest_categorical(
            "initial_agent_timesteps", [10000, 20000, 40000]
        ),
    }
    if uses_preferences(arm):
        params["batch_size_pref"] = trial.suggest_categorical(
            "batch_size_pref", [64, 128, 256]
        )
        if resolved_labels == "binary_bernoulli" and fix_pref_temperature is None:
            params["pref_temperature"] = trial.suggest_float(
                "pref_temperature", 1.0, 50.0, log=True
            )
    if arm.startswith("pref_"):
        params["query_schedule"] = trial.suggest_categorical(
            "query_schedule", ["constant", "hyperbolic", "inverse_quadratic"]
        )
        params["initial_queries"] = trial.suggest_categorical(
            "initial_queries", initial_queries_choices(pref_budget)
        )
        params["fragmenter_type"] = trial.suggest_categorical(
            "fragmenter_type", ["active", "random"]
        )
    elif arm.startswith("hybrid_") and resolved_labels == "binary_bernoulli":
        # Bernoulli labels make the bootstrap size more consequential. Keep
        # schedule and fragmenter fixed so this adds only one search dimension.
        params["initial_queries"] = trial.suggest_categorical(
            "initial_queries", initial_queries_choices(pref_budget)
        )
    if uses_demos(arm):
        params["batch_size_expert"] = trial.suggest_categorical(
            "batch_size_expert", [16, 32, 64, 128]
        )
    if arm.startswith("demo_"):
        params["batch_size_model"] = trial.suggest_categorical(
            "batch_size_model", [32, 64, 128]
        )
    if arm.startswith("hybrid_") and fix_demo_weight is None:
        params["demo_weight"] = trial.suggest_float("demo_weight", 0.1, 10.0, log=True)
    return params


PARAM_TO_OVERRIDE = {
    "lr_rew": "algo.kwargs.lr_rew",
    "gradient_steps_rew": "algo.kwargs.gradient_steps_rew",
    "l2_rew": "algo.kwargs.l2_rew",
    "reward_net_arch": "algo.kwargs.reward_model_kwargs.net_arch",
    "initial_agent_timesteps": "algo.kwargs.initial_agent_timesteps",
    "batch_size_pref": "algo.kwargs.batch_size_pref",
    "query_schedule": "algo.kwargs.query_schedule",
    "initial_queries": "algo.kwargs.initial_queries",
    "fragmenter_type": "algo.kwargs.fragmenter_type",
    "batch_size_expert": "algo.kwargs.batch_size_expert",
    "batch_size_model": "algo.kwargs.batch_size_model",
    "demo_weight": "algo.kwargs.demo_weight",
    "pref_temperature": "algo.kwargs.pref_temperature",
}


def params_to_overrides(params: dict) -> list:
    """Map sampled/best params to Hydra overrides (shared with export_best_config)."""
    return [f"{PARAM_TO_OVERRIDE[name]}={value}" for name, value in params.items()]


def build_command(trial_dir: Path, run_name: str, trial_overrides: list, args) -> list:
    cmd = []
    if args.cores:
        cmd += ["taskset", "-c", args.cores]
    cmd += [
        sys.executable,
        str(TRAIN_SCRIPT),
        f"run.seed={args.seed}",
        f"run.output_dir={trial_dir}",
        f"run.name={run_name}",
        f"run.group=tune_{args.arm}{args.study_suffix}",
        f"wandb.entity={args.wandb_entity}",
        f"wandb.project={args.wandb_project}",
        f"wandb.tags=[optuna,{args.arm}]",
        f"env.n_envs={args.n_envs}",
        f"train.kwargs.total_timesteps={args.total_timesteps}",
        f"train.kwargs.timesteps_per_iteration={args.timesteps_per_iteration}",
        f"eval.n_episodes={args.eval_episodes}",
        *FIXED_OVERRIDES,
        *trial_overrides,
        # Last so they win over both fixed and sampled values (smoke tests,
        # manual pinning).
        *args.override,
    ]
    return cmd


def read_new_jsonl_lines(path: Path, offset: int):
    """Return (records, new_offset) for the lines appended after `offset`."""
    records = []
    with open(path) as f:
        f.seek(offset)
        for line in f:
            if not line.endswith("\n"):  # partial write, retry next poll
                break
            offset += len(line.encode())
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records, offset


def terminate_process_group(process) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(process.pid), sig)
        except (ProcessLookupError, PermissionError):
            return
        deadline = time.time() + KILL_GRACE_SECONDS
        while time.time() < deadline:
            if process.poll() is not None:
                return
            time.sleep(1)


def make_objective(args):
    out_root = Path(args.output_root) / f"hybrid_sac_{args.arm}{args.study_suffix}"

    def objective(trial: optuna.Trial) -> float:
        # First thing, before anything can fail: bind this trial to the worker
        # process running it. Without this the campaign manager can detect that
        # a RUNNING trial is orphaned but not WHICH one.
        if args.worker_token:
            trial.set_user_attr("worker_token", args.worker_token)
        trial_dir = out_root / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        run_name = f"{args.arm}{args.study_suffix}-t{trial.number:03d}"
        params = suggest_params(
            trial,
            args.arm,
            args.pref_budget,
            args.preference_labels,
            fix_demo_weight=args.fix_demo_weight,
            fix_pref_temperature=args.fix_pref_temperature,
        )
        trial_overrides = (
            arm_overrides(
                args.arm,
                args.pref_budget,
                args.demo_budget,
                args.preference_labels,
            )
            + params_to_overrides(params)
            # Last inside trial_overrides so pinned values beat FIXED_OVERRIDES
            # (which sets pref_temperature=20.0); args.override still wins.
            + fixed_param_overrides(
                args.fix_demo_weight, args.fix_pref_temperature
            )
        )
        cmd = build_command(trial_dir, run_name, trial_overrides, args)
        (trial_dir / "command.txt").write_text(" ".join(cmd) + "\n")

        env = {
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        log_file = open(trial_dir / "train.log", "w")
        process = subprocess.Popen(
            cmd, cwd=REPO_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        # The trainer runs in its own session (start_new_session=True), so if
        # this worker dies the trainer survives and keeps holding its core.
        # Record the group ids here so the campaign manager can clean it up;
        # without this record it has no safe way to tell which process to kill.
        runtime_record = {
            "worker_token": args.worker_token,
            "study_name": f"hybrid_sac_{args.arm}{args.study_suffix}",
            "trial_id": trial._trial_id,
            "trial_number": trial.number,
            "run_name": run_name,
            "worker_pid": os.getpid(),
            "trainer_pid": process.pid,
            "trainer_pgid": os.getpgid(process.pid),
            "trainer_sid": os.getsid(process.pid),
            "core": args.cores,
            "started_at": time.time(),
        }
        runtime_path = trial_dir / "trial_runtime.json"
        temporary = runtime_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(runtime_record, indent=2, sort_keys=True) + "\n")
        temporary.replace(runtime_path)

        started = time.time()
        metrics_path, offset, last_step = None, 0, -1
        collapse_streak = 0
        try:
            while True:
                exited = process.poll() is not None
                if metrics_path is None:
                    found = sorted(trial_dir.glob("*/metrics.jsonl"))
                    metrics_path = found[0] if found else None
                if metrics_path is not None:
                    records, offset = read_new_jsonl_lines(metrics_path, offset)
                    for record in records:
                        step = int(record.get("iterations", last_step + 1))
                        value = record.get(PRUNE_METRIC)
                        if value is None or step <= last_step:
                            continue
                        last_step = step
                        trial.report(float(value), step)
                        if trial.should_prune():
                            trial.set_user_attr("prune_reason", "median")
                            terminate_process_group(process)
                            raise optuna.TrialPruned()
                        # Absolute collapse rule (see COLLAPSE_* above).
                        if step >= COLLAPSE_MIN_STEP and float(value) <= COLLAPSE_REWARD:
                            collapse_streak += 1
                            if collapse_streak >= COLLAPSE_STREAK:
                                trial.set_user_attr("prune_reason", "collapse")
                                terminate_process_group(process)
                                raise optuna.TrialPruned()
                        else:
                            collapse_streak = 0
                if exited:
                    break
                if time.time() - started > args.trial_timeout:
                    # PRUNED, not FAIL: a timeout eliminates a too-slow config,
                    # it is not an infrastructure failure. Schedulers gate on
                    # the FAIL ratio, which must measure crashes/OOM/W&B only.
                    trial.set_user_attr("prune_reason", "timeout")
                    terminate_process_group(process)
                    raise optuna.TrialPruned()
                time.sleep(POLL_SECONDS)
        finally:
            log_file.close()

        if process.returncode != 0:
            raise RuntimeError(
                f"Training subprocess failed (exit {process.returncode}); "
                f"see {trial_dir / 'train.log'}."
            )
        eval_files = sorted(trial_dir.glob("*/final_eval.json"))
        if not eval_files:
            raise RuntimeError(f"No final_eval.json produced under {trial_dir}.")
        metrics = json.loads(eval_files[0].read_text())
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("run_dir", str(eval_files[0].parent))
        return float(metrics[OBJECTIVE_METRIC])

    return objective


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--n-trials", type=int, required=True,
                        help="Trials run by THIS worker (budget is shared via the journal).")
    parser.add_argument("--storage-path", default="outputs/optuna/journal.log")
    parser.add_argument("--output-root", default="outputs/optuna")
    parser.add_argument("--cores", default=None,
                        help="taskset CPU list for the training subprocess, e.g. '33-35'.")
    parser.add_argument("--n-envs", type=int, default=2,
                        help="SubprocVecEnv size; must be >=2 (n_envs=1 breaks libsumo).")
    parser.add_argument("--seed", type=int, default=0, help="Training seed for every trial.")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--timesteps-per-iteration", type=int, default=20_000)
    parser.add_argument("--pref-budget", type=int, default=5000,
                        help="total_queries for the pref/hybrid arms.")
    parser.add_argument("--demo-budget", type=int, default=500,
                        help="n_expert_trajectories for the demo/hybrid arms.")
    parser.add_argument(
        "--preference-labels",
        choices=PREFERENCE_LABEL_CHOICES,
        default="auto",
        help=(
            "Preference oracle labels. 'auto' preserves the historical defaults "
            "(Bernoulli only for pref_bernoulli). Selecting binary_bernoulli also "
            "adds pref_temperature to the Optuna search space."
        ),
    )
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--trial-timeout", type=int, default=12 * 3600)
    parser.add_argument("--wandb-entity", default="andrea02polimi-politecnico-di-milano")
    parser.add_argument("--wandb-project", default="tuning-thesis")
    parser.add_argument("--sampler-startup-trials", type=int, default=8,
                        help="Random trials before TPE starts modelling.")
    parser.add_argument("--pruner-warmup-frac", type=float, default=0.4,
                        help="Fraction of iterations before pruning may trigger.")
    parser.add_argument("--pruner-startup-trials", type=int, default=8,
                        help="Completed trials required before pruning may trigger.")
    parser.add_argument("--study-suffix", default="",
                        help="Suffix appended to the study/run names (e.g. '_q100k') to "
                             "start a FRESH study when the arm is re-tuned at a "
                             "different budget. Never mix budgets in one study.")
    parser.add_argument("--enqueue-params", default=None,
                        help="JSON file with a list of param dicts to enqueue as "
                             "warm-start trials (e.g. baseline winners for hybrid arms). "
                             "NOTE: enqueue_trial can duplicate trials when several "
                             "workers call it at once; prefer enqueueing once from a "
                             "coordinator holding a lock.")
    parser.add_argument("--fix-demo-weight", type=float, default=None,
                        help="Pin algo.kwargs.demo_weight and REMOVE it from the "
                             "search space (hybrid arms). Requires --study-suffix.")
    parser.add_argument("--fix-pref-temperature", type=float, default=None,
                        help="Pin algo.kwargs.pref_temperature and REMOVE it from the "
                             "search space (Bernoulli labels). Requires --study-suffix.")
    parser.add_argument("--worker-token", default=None,
                        help="Opaque id stored on the trial as user attr and in "
                             "trial_runtime.json, so a coordinator can bind trial, "
                             "worker process and trainer process group.")
    parser.add_argument("--override", action="append", default=[],
                        help="Extra Hydra override appended last (wins over fixed "
                             "and sampled values). Repeatable.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_envs < 2:
        raise SystemExit("--n-envs must be >= 2 (DummyVecEnv + two envs breaks libsumo).")
    if args.cores and shutil.which("taskset") is None:
        raise SystemExit("--cores requires taskset (Linux). Omit it on macOS.")
    if not uses_preferences(args.arm) and args.preference_labels != "auto":
        raise SystemExit(
            "--preference-labels is only meaningful for pref/hybrid arms."
        )
    default_labels = resolve_preference_labels(args.arm)
    effective_labels = resolve_preference_labels(args.arm, args.preference_labels)
    if effective_labels != default_labels and not args.study_suffix:
        raise SystemExit(
            "A non-default --preference-labels value requires --study-suffix "
            "so incompatible trials cannot be mixed in one Optuna study."
        )
    pins = {
        "demo_weight": args.fix_demo_weight,
        "pref_temperature": args.fix_pref_temperature,
    }
    if any(value is not None for value in pins.values()) and not args.study_suffix:
        raise SystemExit(
            "--fix-demo-weight/--fix-pref-temperature change the search space "
            "and therefore require a dedicated --study-suffix."
        )
    # --override wins last, so an override on a pinned key would silently
    # defeat the pin (and the study's recorded contract).
    pinned_keys = {
        PARAM_TO_OVERRIDE[name] for name, value in pins.items() if value is not None
    }
    for override in args.override:
        key = override.split("=", 1)[0].strip()
        if key in pinned_keys:
            raise SystemExit(
                f"--override {key}=... conflicts with the pinned value for that "
                f"key; drop the override or the --fix-* flag."
            )

    storage_path = Path(args.storage_path)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(
        JournalFileBackend(str(storage_path), lock_obj=JournalFileOpenLock(str(storage_path)))
    )
    n_iterations = args.total_timesteps // args.timesteps_per_iteration
    study = optuna.create_study(
        study_name=f"hybrid_sac_{args.arm}{args.study_suffix}",
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            multivariate=True, n_startup_trials=args.sampler_startup_trials
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=args.pruner_startup_trials,
            n_warmup_steps=int(args.pruner_warmup_frac * n_iterations),
            interval_steps=2,
        ),
        load_if_exists=True,
    )
    stored_labels = study.user_attrs.get("preference_labels")
    if stored_labels is not None and stored_labels != effective_labels:
        raise SystemExit(
            f"Study already uses preference_labels={stored_labels!r}, "
            f"not {effective_labels!r}."
        )
    if stored_labels is None:
        study.set_user_attr("preference_labels", effective_labels)

    # The whole experimental contract, not just the labels: a study must never
    # mix trials drawn under different budgets, pins or search spaces.
    contract = {
        "pref_budget": args.pref_budget,
        "demo_budget": args.demo_budget,
        "preference_labels": effective_labels,
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "timesteps_per_iteration": args.timesteps_per_iteration,
        "n_envs": args.n_envs,
        "eval_episodes": args.eval_episodes,
        "search_space_version": SEARCH_SPACE_VERSION,
        "fixed_params": pins,
    }
    stored_contract = study.user_attrs.get("experiment_contract")
    if stored_contract is not None and stored_contract != contract:
        differences = {
            key: {"study": stored_contract.get(key), "worker": value}
            for key, value in contract.items()
            if stored_contract.get(key) != value
        }
        raise SystemExit(
            "This worker disagrees with the study's recorded contract: "
            + json.dumps(differences, sort_keys=True)
        )
    if stored_contract is None:
        study.set_user_attr("experiment_contract", contract)
        study.set_user_attr("fixed_params", pins)

    if args.enqueue_params:
        for params in json.loads(Path(args.enqueue_params).read_text()):
            study.enqueue_trial(params, skip_if_exists=True)
    study.optimize(make_objective(args), n_trials=args.n_trials)

    try:
        best = study.best_trial
    except ValueError:
        # No COMPLETE trial in the study yet: this worker's trial was pruned
        # or failed and it is the first one. Not an error for the worker.
        print(f"\nNo COMPLETE trial in hybrid_sac_{args.arm}{args.study_suffix} yet.")
        return
    print(f"\nBest trial #{best.number}: {OBJECTIVE_METRIC}={best.value:.3f}")
    print(f"  params: {best.params}")
    print(f"  run_dir: {best.user_attrs.get('run_dir')}")


if __name__ == "__main__":
    main()
