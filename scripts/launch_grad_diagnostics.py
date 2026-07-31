"""Gradient-diagnostics runs for the hybrid arms at homogeneous budgets.

Runs of the hybrid arm at homogeneous budgets (x preference queries AND x
expert trajectories), using the winning configuration of the corresponding
Optuna study. Their point is the ``reward/grad_*`` curves (variance, squared
norms, their ratio, the direction-only variance, and the angle between the
preference and demonstration gradients), which is why they go to their own
W&B project instead of joining an existing one.

Two lanes, selected with ``--lane``:

* ``soft`` — soft oracle labels, study ``_hom_soft`` (best trial #24).
* ``bern`` — Bernoulli labels, study ``_hom_bern`` (best trial #18).

``initial_queries`` follows the 10% convention of the published hybrid curves
(``_hybrid_point`` in schedule_budget_curves_completion.py) in BOTH lanes, so
the two are comparable to each other and to the budget curves whose shape
prompted the investigation. For the Bernoulli lane this deliberately departs
from the tuned value (136 at budget 2723, i.e. 5%).

The demonstration subsample seed is deliberately left unset: it falls back to
the shared ``DEMO_SUBSAMPLE_SEED``, so all seeds at a given budget train on
exactly the same demonstrations, and each run records the fingerprint that
proves it (``demo_subsample.json``; check with verify_demo_subsample.py).

Usage::

    python scripts/launch_grad_diagnostics.py --lane bern --dry-run
    python scripts/launch_grad_diagnostics.py --lane bern
    python scripts/launch_grad_diagnostics.py --lane bern --status
"""

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path("/home/fis3/miniconda3/envs/sumo-rlhf/bin/python")
EXPORT_SCRIPT = REPO_ROOT / "scripts" / "export_best_config.py"
JOURNAL = REPO_ROOT / "outputs" / "optuna" / "journal.log"

OUTPUT_ROOT = REPO_ROOT / "outputs" / "grad_diagnostics"
LOG_ROOT = OUTPUT_ROOT / "logs"

WANDB_ENTITY = "andrea02polimi-politecnico-di-milano"
WANDB_PROJECT = "thesis-grad-diagnostics"

SEEDS = (1, 2, 3)
TOTAL_TIMESTEPS = 2_000_000
TIMESTEPS_PER_ITERATION = 20_000
N_ENVS = 2
EVAL_EPISODES = 20
DEMO_WEIGHT = 1.0

# One core per run, inside the half of the machine reserved for this user.
# Lanes launched while another lane is still running must not share cores, so
# each lane takes its own slice.
SLOTS = {
    "soft": tuple(str(core) for core in range(24, 33)),
    "bern": tuple(str(core) for core in range(33, 42)),
    "demo2": tuple(str(core) for core in range(42, 48)),
}

LANES = {
    "soft": {
        "arm": "hybrid_demo_2",
        "preferences": True,
        "study_suffix": "_hom_soft",
        "preference_labels": "soft",
        "curve_arm": "hybrid_demo_2_soft_hom",
        "pref_temperature": 20.0,
        "budgets": (10, 100, 1000),
        "initial_queries": {10: 1, 100: 10, 1000: 100},
        # Tuned values that must survive the budget overrides (trial #24).
        "normalize_agent_reward": True,
        "tuned": {"gradient_steps_rew": 139, "batch_size_pref": 256, "batch_size_expert": 16},
    },
    "bern": {
        "arm": "hybrid_demo_2",
        "preferences": True,
        "study_suffix": "_hom_bern",
        "preference_labels": "binary_bernoulli",
        "curve_arm": "hybrid_demo_2_bern_hom",
        # The Bernoulli oracle temperature, not the soft lane's 20.0.
        "pref_temperature": 3.0595414013726767,
        "budgets": (1000, 2723),
        "initial_queries": {1000: 100, 2723: 272},
        # Tuned values that must survive the budget overrides (trial #18).
        "normalize_agent_reward": True,
        "tuned": {"gradient_steps_rew": 78, "batch_size_pref": 256, "batch_size_expert": 64},
    },
    # Control for the hybrid collapse: demonstrations ONLY, on exactly the
    # demonstrations hybrid soft sees at the same budget (same shared subsample
    # seed => same fingerprint). Isolates the effect of adding the preference
    # channel, since everything else about the demo signal is identical.
    "demo2": {
        "arm": "demo_2",
        "preferences": False,
        "study_suffix": "_no_norm",
        "preference_labels": "auto",
        "curve_arm": "demo_2_no_norm",
        "pref_temperature": None,
        "budgets": (10,),
        "initial_queries": {10: 0},
        # The _no_norm study was tuned with the agent-facing reward
        # normalization OFF, passed as an extra override by the campaign
        # script; export_best_config's fixed block emits true, so it must be
        # restated here or this control would not match the published demo_2.
        "normalize_agent_reward": False,
        # Tuned values that must survive the budget overrides (trial #26).
        "tuned": {"gradient_steps_rew": 100, "batch_size_expert": 16, "batch_size_model": 64},
    },
}


def manifest_path(lane: str) -> Path:
    return OUTPUT_ROOT / f"manifest_{lane}.json"


@dataclass(frozen=True)
class Task:
    lane: str
    budget: int
    seed: int

    @property
    def group(self) -> str:
        return f"gd_{LANES[self.lane]['curve_arm']}_B{self.budget}"

    @property
    def run_name(self) -> str:
        return f"{self.group}-seed{self.seed}"

    @property
    def initial_queries(self) -> int:
        return LANES[self.lane]["initial_queries"][self.budget]

    @property
    def output_dir(self) -> Path:
        return OUTPUT_ROOT / self.group

    @property
    def log_path(self) -> Path:
        return LOG_ROOT / f"{self.run_name}.log"


def build_tasks(lane: str) -> tuple[Task, ...]:
    return tuple(
        Task(lane, budget, seed)
        for budget in LANES[lane]["budgets"]
        for seed in SEEDS
    )


def export_overrides(lane: str, budget: int) -> tuple[str, ...]:
    """Tuned hyperparameters of the winning trial, as Hydra overrides."""
    config = LANES[lane]
    command = [
        str(PYTHON), str(EXPORT_SCRIPT),
        "--arm", config["arm"],
        "--study-suffix", config["study_suffix"],
        "--format", "full",
        "--storage-path", str(JOURNAL.relative_to(REPO_ROOT)),
        "--pref-budget", str(budget),
        "--demo-budget", str(budget),
        "--preference-labels", config["preference_labels"],
    ]
    result = subprocess.run(
        command, cwd=REPO_ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    overrides = tuple(shlex.split(result.stdout.strip()))
    if not overrides:
        raise RuntimeError(f"No overrides exported for {config['arm']}{config['study_suffix']}.")
    return overrides


def task_overrides(task: Task) -> tuple[str, ...]:
    """Per-run overrides. These come last, so Hydra takes them over the export.

    ``initial_queries`` in particular MUST be re-stated: the tuner's arm
    preset computes ``max(100, 0.1 * pref_budget)`` and the Bernoulli lane
    also carries its own tuned value, neither of which is what we want here.
    """
    config = LANES[task.lane]
    queries = task.budget if config["preferences"] else 0
    preference_overrides = (
        (f"algo.kwargs.labels_type={config['preference_labels']}",
         f"algo.kwargs.pref_temperature={config['pref_temperature']}")
        if config["preferences"] else ()
    )
    return (
        f"algo.kwargs.total_queries={queries}",
        f"train.kwargs.total_queries={queries}",
        f"algo.kwargs.initial_queries={task.initial_queries}",
        f"run.n_expert_trajectories={task.budget}",
        # demo_subsample_seed intentionally omitted: the config default is the
        # shared constant, so every arm and seed reads the same demonstrations.
        *preference_overrides,
        f"algo.kwargs.demo_weight={DEMO_WEIGHT}",
        f"algo.kwargs.normalize_agent_reward="
        f"{str(config['normalize_agent_reward']).lower()}",
        f"run.seed={task.seed}",
        f"run.output_dir={task.output_dir.relative_to(REPO_ROOT)}",
        f"run.name={task.run_name}",
        f"run.group={task.group}",
        f"wandb.entity={WANDB_ENTITY}",
        f"wandb.project={WANDB_PROJECT}",
        f"wandb.tags=[grad_diagnostics,{config['curve_arm']},hom,B{task.budget}]",
        f"env.n_envs={N_ENVS}",
        f"eval.n_episodes={EVAL_EPISODES}",
        f"train.kwargs.total_timesteps={TOTAL_TIMESTEPS}",
        f"train.kwargs.timesteps_per_iteration={TIMESTEPS_PER_ITERATION}",
    )


def full_command(task: Task, base: tuple[str, ...]) -> list[str]:
    return [str(PYTHON), "scripts/train_hybrid_sac.py", *base, *task_overrides(task)]


def validate(lane: str, tasks: tuple[Task, ...], exports: dict) -> list[dict]:
    """Resolve each budget's config with Hydra and check what actually lands.

    Catches a bad override before six hours of compute, not after.
    """
    from omegaconf import OmegaConf

    config = LANES[lane]
    records = []
    seen: set[int] = set()
    for task in tasks:
        if task.budget in seen:
            continue
        seen.add(task.budget)
        command = full_command(task, exports[task.budget])
        command[2:2] = ["--cfg", "job", "--resolve"]
        result = subprocess.run(
            command, cwd=REPO_ROOT, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        cfg = OmegaConf.create(result.stdout)
        checks = {
            "algo.total_queries": (cfg.algo.kwargs.total_queries,
                                   task.budget if config["preferences"] else 0),
            "train.total_queries": (cfg.train.kwargs.total_queries,
                                    task.budget if config["preferences"] else 0),
            "n_expert_trajectories": (cfg.run.n_expert_trajectories, task.budget),
            "initial_queries": (cfg.algo.kwargs.initial_queries, task.initial_queries),
            "demo_subsample_seed": (cfg.run.demo_subsample_seed, 1000),
            "n_expert_transitions": (cfg.run.n_expert_transitions, None),
            "demo_mode": (cfg.algo.kwargs.demo_mode, "gcl"),
            "loss_type": (cfg.algo.kwargs.loss_type, "demo_2"),
            "demo_weight": (cfg.algo.kwargs.demo_weight, DEMO_WEIGHT),
            "normalize_agent_reward": (cfg.algo.kwargs.normalize_agent_reward,
                                       config["normalize_agent_reward"]),
            "grad_diagnostics_interval": (cfg.algo.kwargs.grad_diagnostics_interval, 1),
            "total_timesteps": (cfg.train.kwargs.total_timesteps, TOTAL_TIMESTEPS),
            "wandb.project": (cfg.wandb.project, WANDB_PROJECT),
        }
        if config["preferences"]:
            checks["labels_type"] = (cfg.algo.kwargs.labels_type,
                                     config["preference_labels"])
            checks["pref_temperature"] = (cfg.algo.kwargs.pref_temperature,
                                          config["pref_temperature"])
        for name, expected in config["tuned"].items():
            checks[name] = (cfg.algo.kwargs[name], expected)
        bad = {k: v for k, v in checks.items() if v[0] != v[1]}
        if bad:
            raise RuntimeError(f"Config mismatch at budget {task.budget}: {bad}")
        if not 0 <= cfg.algo.kwargs.initial_queries <= cfg.algo.kwargs.total_queries:
            raise RuntimeError(f"initial_queries out of range at budget {task.budget}.")
        records.append({"budget": task.budget, "checks": {k: v[0] for k, v in checks.items()}})
    return records


def launch(lane: str, tasks: tuple[Task, ...], exports: dict) -> list[dict]:
    slots = SLOTS[lane]
    if len(tasks) > len(slots):
        raise RuntimeError(f"{len(tasks)} tasks but only {len(slots)} slots for lane {lane}.")
    launched = []
    for task, slot in zip(tasks, slots):
        task.output_dir.mkdir(parents=True, exist_ok=True)
        task.log_path.parent.mkdir(parents=True, exist_ok=True)
        command = ["taskset", "-c", slot, *full_command(task, exports[task.budget])]
        with open(task.log_path, "w") as log:
            process = subprocess.Popen(
                command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,  # survives the launching shell
            )
        launched.append({
            "run_name": task.run_name,
            "group": task.group,
            "budget": task.budget,
            "seed": task.seed,
            "initial_queries": task.initial_queries,
            "pid": process.pid,
            "cpu": slot,
            "log": str(task.log_path.relative_to(REPO_ROOT)),
        })
        print(f"launched {task.run_name} pid={process.pid} cpu={slot}")
    return launched


def status(lane: str) -> int:
    path = manifest_path(lane)
    if not path.exists():
        print(f"No manifest for lane {lane}: nothing launched yet.")
        return 1
    manifest = json.loads(path.read_text())
    print(f"lane {lane} started {manifest['started']} -> {manifest['wandb_project']}\n")
    header = f"{'run':<40} {'pid':>8}  {'state':<9} {'iter':>6}  fingerprint"
    print(header)
    print("-" * len(header))
    for entry in manifest["runs"]:
        alive = Path(f"/proc/{entry['pid']}").exists()
        run_dir = OUTPUT_ROOT / entry["group"] / entry["run_name"]
        final = run_dir / "final_eval.json"
        state = "done" if final.exists() else ("running" if alive else "STOPPED")
        metrics = run_dir / "metrics.jsonl"
        iterations = sum(1 for _ in metrics.open()) if metrics.exists() else 0
        subsample = run_dir / "demo_subsample.json"
        fingerprint = "-"
        if subsample.exists():
            fingerprint = json.loads(subsample.read_text())["fingerprint"][:12]
        print(f"{entry['run_name']:<40} {entry['pid']:>8}  {state:<9} {iterations:>6}  {fingerprint}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lane", choices=sorted(LANES), required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="validate the resolved configs and print a command")
    parser.add_argument("--status", action="store_true",
                        help="report on the runs recorded in the lane's manifest")
    args = parser.parse_args()

    if args.status:
        return status(args.lane)

    lane = args.lane
    tasks = build_tasks(lane)
    exports = {budget: export_overrides(lane, budget) for budget in LANES[lane]["budgets"]}
    records = validate(lane, tasks, exports)

    print(f"lane {lane}: {len(tasks)} runs, budgets {LANES[lane]['budgets']} x seeds {SEEDS}")
    for record in records:
        checks = record["checks"]
        print(
            f"  budget {record['budget']:>4}: queries={checks['algo.total_queries']} "
            f"demos={checks['n_expert_trajectories']} "
            f"initial_queries={checks['initial_queries']} "
            f"labels={checks.get('labels_type', 'n/d')} "
            f"subsample_seed={checks['demo_subsample_seed']}"
        )
    print(f"project {WANDB_PROJECT}, {TOTAL_TIMESTEPS} timesteps each\n")

    if args.dry_run:
        print(shlex.join(full_command(tasks[0], exports[tasks[0].budget])))
        return 0

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    launched = launch(lane, tasks, exports)
    manifest_path(lane).write_text(json.dumps({
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lane": lane,
        "wandb_project": WANDB_PROJECT,
        "study": f"hybrid_sac_{LANES[lane]['arm']}{LANES[lane]['study_suffix']}",
        "total_timesteps": TOTAL_TIMESTEPS,
        "runs": launched,
    }, indent=2))
    print(f"\nmanifest: {manifest_path(lane).relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
