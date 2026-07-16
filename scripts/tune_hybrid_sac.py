"""Optuna hyperparameter search for the HybridAlgorithm SAC baseline arms.

Four arms, one Optuna study each (sharing one journal file):

* ``pref_soft``      — preferences only, soft oracle labels
* ``pref_bernoulli`` — preferences only, sampled binary oracle labels
* ``demo_demo``      — demonstrations only, difference-of-means loss
* ``demo_maxent2``   — demonstrations only, MaxEnt-2 loss

Each trial runs ``scripts/test_hybrid_SAC.py`` as a subprocess (libsumo, W&B
and SubprocVecEnv state are all cleaned up by process exit), follows the
per-iteration ``metrics.jsonl`` written by the training run for median
pruning, and reads the final held-out ``final_eval.json`` as the objective
(``eval/mean_fast_return``, maximized).

Multiple workers can run in parallel against the same ``--storage-path``
(JournalStorage); pin each worker with ``--cores`` on Linux (taskset).
Every override is passed explicitly so trials never depend on yaml/launcher
default drift.
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
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "test_hybrid_SAC.py"

ARMS = ("pref_soft", "pref_bernoulli", "demo_demo", "demo_maxent2")
PRUNE_METRIC = "rollout/mean_true_reward"
OBJECTIVE_METRIC = "eval/mean_fast_return"
POLL_SECONDS = 30
KILL_GRACE_SECONDS = 30

# Values fixed across all arms and trials. l2_rew MUST stay at 1e-4 (1e-2
# collapses the reward net, diagnosed 2026-07-05); pref_temperature and
# preference_fragment_length define the synthetic oracle, i.e. the problem,
# not the learner. SAC kwargs are the tuned launcher values.
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
    "agent.kwargs.gradient_steps=64",
    "agent.kwargs.policy_kwargs.net_arch=[64,64]",
    "agent.kwargs.device=cpu",
    "algo.kwargs.demo_mode=gcl",
    "algo.kwargs.relabel_rewards=true",
    "algo.kwargs.normalize_agent_reward=true",
    "algo.kwargs.l2_rew=0.0001",
    "algo.kwargs.temperature=1.0",
    "algo.kwargs.pref_temperature=20.0",
    "algo.kwargs.preference_fragment_length=1",
    "algo.kwargs.fragmenter_type=active",
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


def arm_overrides(arm: str, args) -> list:
    """Arm-defining overrides (the launcher MODE presets, made explicit)."""
    if arm.startswith("pref_"):
        labels = "soft" if arm == "pref_soft" else "binary_bernoulli"
        return [
            "algo.kwargs.demo_weight=0.0",
            "algo.kwargs.loss_type=maxent_2",  # inert with demo_weight=0
            f"algo.kwargs.labels_type={labels}",
            f"algo.kwargs.total_queries={args.pref_budget}",
            f"train.kwargs.total_queries={args.pref_budget}",
            # Pref arms use the yaml reward net; demo arms search it.
            "algo.kwargs.reward_model_kwargs.net_arch=[128,128]",
        ]
    loss = "demo" if arm == "demo_demo" else "maxent_2"
    return [
        "algo.kwargs.demo_weight=1.0",
        f"algo.kwargs.loss_type={loss}",
        "algo.kwargs.labels_type=soft",  # inert with zero queries
        "algo.kwargs.total_queries=0",
        "algo.kwargs.initial_queries=0",
        "train.kwargs.total_queries=0",
        "algo.kwargs.query_schedule=constant",
        "algo.kwargs.batch_size_pref=128",  # inert with zero queries
        f"run.n_expert_trajectories={args.demo_budget}",
    ]


def suggest_overrides(trial: optuna.Trial, arm: str, args) -> list:
    """Trial-sampled overrides: the per-arm search space."""
    lr_rew = trial.suggest_float("lr_rew", 3e-5, 3e-3, log=True)
    gradient_steps_rew = trial.suggest_int("gradient_steps_rew", 50, 400, log=True)
    initial_agent_timesteps = trial.suggest_categorical(
        "initial_agent_timesteps", [10000, 20000, 40000]
    )
    overrides = [
        f"algo.kwargs.lr_rew={lr_rew}",
        f"algo.kwargs.gradient_steps_rew={gradient_steps_rew}",
        f"algo.kwargs.initial_agent_timesteps={initial_agent_timesteps}",
    ]
    if arm.startswith("pref_"):
        batch_size_pref = trial.suggest_categorical("batch_size_pref", [64, 128, 256])
        query_schedule = trial.suggest_categorical(
            "query_schedule", ["constant", "hyperbolic", "inverse_quadratic"]
        )
        initial_queries = trial.suggest_categorical(
            "initial_queries", [100, 250, 500, 1000]
        )
        overrides += [
            f"algo.kwargs.batch_size_pref={batch_size_pref}",
            f"algo.kwargs.query_schedule={query_schedule}",
            f"algo.kwargs.initial_queries={initial_queries}",
            # Inert with demo_weight=0, pinned for a clean config record.
            "algo.kwargs.batch_size_expert=64",
            "algo.kwargs.batch_size_model=64",
        ]
    else:
        batch_size_expert = trial.suggest_categorical(
            "batch_size_expert", [16, 32, 64, 128]
        )
        batch_size_model = trial.suggest_categorical("batch_size_model", [32, 64, 128])
        net_arch = trial.suggest_categorical("reward_net_arch", ["[64,64]", "[128,128]"])
        overrides += [
            f"algo.kwargs.batch_size_expert={batch_size_expert}",
            f"algo.kwargs.batch_size_model={batch_size_model}",
            f"algo.kwargs.reward_model_kwargs.net_arch={net_arch}",
        ]
    return overrides


def build_command(trial_dir: Path, trial_overrides: list, args) -> list:
    cmd = []
    if args.cores:
        cmd += ["taskset", "-c", args.cores]
    cmd += [
        sys.executable,
        str(TRAIN_SCRIPT),
        f"run.seed={args.seed}",
        f"run.output_dir={trial_dir}",
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
    out_root = Path(args.output_root) / f"hybrid_sac_{args.arm}"

    def objective(trial: optuna.Trial) -> float:
        trial_dir = out_root / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial_overrides = arm_overrides(args.arm, args) + suggest_overrides(
            trial, args.arm, args
        )
        cmd = build_command(trial_dir, trial_overrides, args)
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
        started = time.time()
        metrics_path, offset, last_step = None, 0, -1
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
                            terminate_process_group(process)
                            raise optuna.TrialPruned()
                if exited:
                    break
                if time.time() - started > args.trial_timeout:
                    terminate_process_group(process)
                    raise RuntimeError(
                        f"Trial exceeded --trial-timeout={args.trial_timeout}s."
                    )
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
                        help="total_queries for the pref arms.")
    parser.add_argument("--demo-budget", type=int, default=500,
                        help="n_expert_trajectories for the demo arms.")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--trial-timeout", type=int, default=12 * 3600)
    parser.add_argument("--wandb-entity", default="andrea02polimi-politecnico-di-milano")
    parser.add_argument("--wandb-project", default="preference+demonstration")
    parser.add_argument("--pruner-warmup-frac", type=float, default=0.4,
                        help="Fraction of iterations before pruning may trigger.")
    parser.add_argument("--pruner-startup-trials", type=int, default=8,
                        help="Completed trials required before pruning may trigger.")
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

    storage_path = Path(args.storage_path)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(
        JournalFileBackend(str(storage_path), lock_obj=JournalFileOpenLock(str(storage_path)))
    )
    n_iterations = args.total_timesteps // args.timesteps_per_iteration
    study = optuna.create_study(
        study_name=f"hybrid_sac_{args.arm}",
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(multivariate=True, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=args.pruner_startup_trials,
            n_warmup_steps=int(args.pruner_warmup_frac * n_iterations),
            interval_steps=2,
        ),
        load_if_exists=True,
    )
    study.optimize(make_objective(args), n_trials=args.n_trials)

    best = study.best_trial
    print(f"\nBest trial #{best.number}: {OBJECTIVE_METRIC}={best.value:.3f}")
    print(f"  params: {best.params}")
    print(f"  run_dir: {best.user_attrs.get('run_dir')}")


if __name__ == "__main__":
    main()
