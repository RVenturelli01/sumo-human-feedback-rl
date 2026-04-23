"""
Optuna hyperparameter search for ChristianoAlgorithm.

The study is persisted to a SQLite DB so trials can be interrupted
and resumed without losing progress. Re-run the same command to resume.

Usage:
    python scripts/tune_christiano.py --no-wandb                     # local test
    python scripts/tune_christiano.py --n-trials 50                  # full run
    python scripts/tune_christiano.py --n-trials 20 --storage sqlite:////storage/fis3/optuna.db
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import wandb
from stable_baselines3 import PPO
from sumo_gym_ego import EgoStatus

import sumo_rl_ego as sre
from human_feedback_rl.algorithms import ChristianoAlgorithm


# ── logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("tune")


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(agent, n_episodes: int, seed: int) -> float:
    """Return success rate of `agent` over `n_episodes` single-env episodes."""
    eval_env = sre.make_env("HighwayEgo-v0", seed=seed, ego="continuous", reward="fast")
    policy = sre.ModelPolicy(agent)
    successes = 0
    try:
        for ep in range(n_episodes):
            info = sre.run_episode(eval_env, policy, seed=seed + ep)
            if info.get("ego_status") == EgoStatus.ARRIVED.value:
                successes += 1
    finally:
        eval_env.close()
    return successes / n_episodes


# ── callback: salva il best dopo ogni trial ───────────────────────────────────

def make_checkpoint_callback(output_dir: Path, log: logging.Logger):
    best_path   = output_dir / "best_params.json"
    trials_path = output_dir / "all_trials.json"

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        # aggiorna all_trials.json con tutti i completed
        completed = [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        trials_path.write_text(json.dumps(completed, indent=2))

        # aggiorna best_params.json se questo trial è il nuovo best
        if trial.state == optuna.trial.TrialState.COMPLETE:
            if study.best_trial.number == trial.number:
                best = {"value": trial.value, "params": trial.params}
                best_path.write_text(json.dumps(best, indent=2))
                log.info(
                    f"Trial {trial.number} nuovo best → success_rate={trial.value:.3f}"
                )
            else:
                log.info(
                    f"Trial {trial.number} completato → success_rate={trial.value:.3f} "
                    f"(best={study.best_value:.3f})"
                )
        elif trial.state == optuna.trial.TrialState.PRUNED:
            log.warning(f"Trial {trial.number} pruned/fallito")

    return callback


# ── objective ─────────────────────────────────────────────────────────────────

def make_objective(args, log: logging.Logger):
    def objective(trial: optuna.Trial) -> float:

        # ── suggest hyperparameters ──────────────────────────────────────────
        lr_rew          = trial.suggest_float("lr_rew",          1e-5, 1e-2, log=True)
        batch_size_rew  = trial.suggest_categorical("batch_size_rew",  [32, 64, 128])
        n_ephochs_rew   = trial.suggest_int("n_ephochs_rew",     3, 20)
        n_ensembles_rew = trial.suggest_int("n_ensembles_rew",   2, 5)
        fragment_length = trial.suggest_categorical("fragment_length", [10, 25, 50])
        query_schedule  = trial.suggest_categorical("query_schedule",
                              ["constant", "hyperbolic", "inverse_quadratic"])
        lr_agent        = trial.suggest_float("lr_agent",        1e-4, 1e-3, log=True)
        ent_coef        = trial.suggest_float("ent_coef",        0.0,  0.1)

        log.info(f"Trial {trial.number} start — params: {dict(trial.params)}")

        run = None
        if args.wandb:
            run = wandb.init(
                project="christiano-optuna",
                group=args.study_name,
                name=f"trial-{trial.number}",
                config=dict(trial.params),
                reinit=True,
            )

        env = sre.make_vec_env(
            "HighwayEgo-v0",
            n_envs=1,
            base_seed=args.seed,
            ego="continuous",
            reward="fast",
        )

        try:
            agent = PPO(
                policy="MlpPolicy",
                env=env,
                learning_rate=lr_agent,
                n_steps=256,
                batch_size=256,
                n_epochs=5,
                gamma=0.995,
                ent_coef=ent_coef,
                seed=args.seed,
                device="cpu",
                verbose=0,
            )

            algo = ChristianoAlgorithm(
                env=env,
                agent=agent,
                lr_rew=lr_rew,
                batch_size_rew=batch_size_rew,
                n_ephochs_rew=n_ephochs_rew,
                n_ensembles_rew=n_ensembles_rew,
                fragment_length=fragment_length,
                query_schedule=query_schedule,
                n_iterations=args.n_iterations,
                rng=np.random.default_rng(args.seed),
            )

            algo.train(
                total_timesteps=args.total_timesteps,
                total_comparisons=args.total_comparisons,
            )

            trained_agent = algo.trajectory_generator.agent
            success_rate = evaluate(trained_agent, n_episodes=args.n_eval_episodes, seed=args.seed)

            if run is not None:
                wandb.log({"success_rate": success_rate})
                run.finish()

            return success_rate

        except Exception:
            log.exception(f"Trial {trial.number} crashed")
            if run is not None:
                run.finish()
            raise optuna.exceptions.TrialPruned()

        finally:
            env.close()

    return objective


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Optuna HPO for ChristianoAlgorithm")
    parser.add_argument("--n-trials",          type=int,  default=30)
    parser.add_argument("--study-name",        type=str,  default="christiano-optuna")
    parser.add_argument("--storage",           type=str,  default=None,
                        help="Optuna storage URL (default: sqlite:///<output-dir>/optuna.db). "
                             "Esempio server: sqlite:////storage/fis3/optuna.db")
    parser.add_argument("--output-dir",        type=str,  default="outputs/optuna",
                        help="Directory per log, best_params.json, all_trials.json")
    parser.add_argument("--seed",              type=int,  default=0)
    parser.add_argument("--n-iterations",      type=int,  default=5,
                        help="Iterazioni ChristianoAlgorithm per trial")
    parser.add_argument("--total-timesteps",   type=int,  default=30_000)
    parser.add_argument("--total-comparisons", type=int,  default=200)
    parser.add_argument("--n-eval-episodes",   type=int,  default=20)
    parser.add_argument("--no-wandb",          action="store_true")
    args = parser.parse_args()
    args.wandb = not args.no_wandb

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = setup_logging(output_dir / f"tune_{timestamp}.log")

    storage = args.storage or f"sqlite:///{output_dir / 'optuna.db'}"

    log.info(f"Study name : {args.study_name}")
    log.info(f"Storage    : {storage}")
    log.info(f"Output dir : {output_dir}")
    log.info(f"Trials     : {args.n_trials} × "
             f"{args.total_timesteps} timesteps / {args.total_comparisons} comparisons")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    already_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    log.info(f"Trial già completati nel DB: {already_done}")

    study.optimize(
        make_objective(args, log),
        n_trials=args.n_trials,
        callbacks=[make_checkpoint_callback(output_dir, log)],
        show_progress_bar=False,  # incompatibile con logging su file
    )

    best = study.best_trial
    log.info("=" * 50)
    log.info(f"BEST trial #{best.number}  success_rate={best.value:.3f}")
    for k, v in best.params.items():
        log.info(f"  {k}: {v}")
    log.info("=" * 50)
    log.info(f"Risultati salvati in: {output_dir}")


if __name__ == "__main__":
    main()