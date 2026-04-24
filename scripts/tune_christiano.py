"""
Optuna multi-objective hyperparameter search for ChristianoAlgorithm.

Obiettivi (in ordine di priorità):
  1. fast_return    — media del ritorno fast sull'episodio
  2. true_mean_rew  — fast_return normalizzato per lunghezza episodio (per-step)
  3. success_rate   — frazione di episodi con EgoStatus.ARRIVED

Il risultato è una Pareto front: nessun trial domina gli altri su tutti e 3 gli assi.
Per scegliere il best, ordina per fast_return (priorità 1) o usa i pesi in best_trials.json.

Il DB SQLite persiste lo study: riesegui lo stesso comando per riprendere.

Usage:
    python scripts/tune_christiano.py --no-wandb
    python scripts/tune_christiano.py --n-trials 60
    python scripts/tune_christiano.py --storage sqlite:////storage/fis3/optuna.db --n-trials 60
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

def evaluate(agent, n_episodes: int, seed: int) -> tuple[float, float, float]:
    """
    Ritorna (mean_fast_return, mean_true_reward_per_step, success_rate).

    - mean_fast_return    : media del ritorno fast sull'episodio
    - mean_true_reward    : fast_return / ep_length  (per-step, non dipende dalla durata)
    - success_rate        : frazione di episodi con EgoStatus.ARRIVED
    """
    eval_env = sre.make_env("HighwayEgo-v0", seed=seed, ego="continuous", reward="fast")
    policy = sre.ModelPolicy(agent)

    fast_returns: list[float] = []
    ep_lengths:   list[int]   = []
    successes:    list[int]   = []

    try:
        for ep in range(n_episodes):
            info = sre.run_episode(eval_env, policy, seed=seed + ep)
            ep_metrics = info.get("metrics", {}).get("episode", {})

            fast_returns.append(float(ep_metrics.get("rewards/ep_fast_return", 0.0)))
            ep_lengths.append(max(1, int(info.get("step", 1))))
            successes.append(int(info.get("ego_status") == EgoStatus.ARRIVED.value))
    finally:
        eval_env.close()

    mean_fast_return = float(np.mean(fast_returns))
    mean_true_reward = float(np.mean(
        [r / l for r, l in zip(fast_returns, ep_lengths)]
    ))
    success_rate = float(np.mean(successes))

    return mean_fast_return, mean_true_reward, success_rate


# ── callback: salva Pareto front dopo ogni trial ──────────────────────────────

def make_checkpoint_callback(output_dir: Path, log: logging.Logger):
    pareto_path = output_dir / "pareto_front2.json"
    trials_path = output_dir / "all_trials2.json"

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        completed = [
            {
                "number": t.number,
                "values": {
                    "fast_return":   t.values[0],
                    "true_mean_rew": t.values[1],
                    "success_rate":  t.values[2],
                },
                "params": t.params,
            }
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        trials_path.write_text(json.dumps(completed, indent=2))

        pareto = [
            {
                "number": t.number,
                "values": {
                    "fast_return":   t.values[0],
                    "true_mean_rew": t.values[1],
                    "success_rate":  t.values[2],
                },
                "params": t.params,
            }
            for t in study.best_trials
        ]
        # ordina per fast_return decrescente (priorità 1)
        pareto.sort(key=lambda x: x["values"]["fast_return"], reverse=True)
        pareto_path.write_text(json.dumps(pareto, indent=2))

        if trial.state == optuna.trial.TrialState.COMPLETE:
            v = trial.values
            log.info(
                f"Trial {trial.number:3d} completato — "
                f"fast_return={v[0]:.3f}  true_mean_rew={v[1]:.4f}  success_rate={v[2]:.3f}  "
                f"| Pareto front: {len(study.best_trials)} trial(s)"
            )
        elif trial.state == optuna.trial.TrialState.PRUNED:
            log.warning(f"Trial {trial.number} pruned/fallito")

    return callback


# ── objective ─────────────────────────────────────────────────────────────────

def make_objective(args, log: logging.Logger):
    def objective(trial: optuna.Trial) -> tuple[float, float, float]:

        # ── suggest hyperparameters ──────────────────────────────────────────
        lr_rew          = trial.suggest_float("lr_rew",          1e-5, 1e-2, log=True)
        batch_size_rew  = trial.suggest_categorical("batch_size_rew",  [64, 128, 256])
        n_ephochs_rew   = trial.suggest_int("n_ephochs_rew",   20, 100, 200)
        n_ensembles_rew = trial.suggest_int("n_ensembles_rew",   2, 3, 5)
        fragment_length = trial.suggest_categorical("fragment_length", [1, 10, 25, 50])
        query_schedule  = trial.suggest_categorical("query_schedule",
                              ["constant", "hyperbolic", "inverse_quadratic"])
        lr_agent        = trial.suggest_float("lr_agent",        1e-4, 1e-3, log=True)
        ent_coef        = trial.suggest_float("ent_coef",        0.0,  0.1)

        log.info(f"Trial {trial.number} start — {dict(trial.params)}")

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
                train_comparison_frac=0.2,
                initial_comparison_frac=0.1,
                initial_epoch_multiplier=2.0,
                rng=np.random.default_rng(args.seed),
            )

            algo.train(
                total_timesteps=args.total_timesteps,
                total_comparisons=args.total_comparisons,
            )

            trained_agent = algo.trajectory_generator.agent
            fast_return, true_mean_rew, success_rate = evaluate(
                trained_agent, n_episodes=args.n_eval_episodes, seed=args.seed
            )

            if run is not None:
                wandb.log({
                    "fast_return":   fast_return,
                    "true_mean_rew": true_mean_rew,
                    "success_rate":  success_rate,
                })
                run.finish()

            return fast_return, true_mean_rew, success_rate

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
    parser = argparse.ArgumentParser(description="Optuna multi-obj HPO for ChristianoAlgorithm")
    parser.add_argument("--n-trials",          type=int,  default=60,
                        help="NSGA-II: i primi population_size trial sono casuali, "
                             "quelli successivi sono ottimizzati")
    parser.add_argument("--population-size",   type=int,  default=30,
                        help="Dimensione della popolazione NSGA-II. "
                             "Devono essere completati prima di iniziare l'ottimizzazione")
    parser.add_argument("--study-name",        type=str,  default="christiano-optuna-mo")
    parser.add_argument("--storage",           type=str,  default=None,
                        help="Optuna storage URL (default: sqlite:///<output-dir>/optuna.db). "
                             "Esempio server: sqlite:////storage/fis3/optuna.db")
    parser.add_argument("--output-dir",        type=str,  default="outputs/optuna",
                        help="Directory per log, pareto_front.json, all_trials.json")
    parser.add_argument("--seed",              type=int,  default=0)
    parser.add_argument("--n-iterations",      type=int,  default=10)
    parser.add_argument("--total-timesteps",   type=int,  default=200_000)
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

    log.info(f"Study        : {args.study_name}")
    log.info(f"Storage      : {storage}")
    log.info(f"Output dir   : {output_dir}")
    log.info(f"Objectives   : fast_return (1°) | true_mean_rew (2°) | success_rate (3°)")
    log.info(f"Trials       : {args.n_trials}  |  population_size={args.population_size}")
    log.info(f"Budget/trial : {args.total_timesteps} ts / {args.total_comparisons} cmp / "
             f"{args.n_eval_episodes} eval ep")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        directions=["maximize", "maximize", "maximize"],
        load_if_exists=True,
        sampler=optuna.samplers.NSGAIISampler(
            population_size=args.population_size,
            seed=args.seed,
        ),
    )

    already_done = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    log.info(f"Trial già completati nel DB: {already_done}")

    study.optimize(
        make_objective(args, log),
        n_trials=args.n_trials,
        callbacks=[make_checkpoint_callback(output_dir, log)],
        show_progress_bar=False,
    )

    # ── stampa Pareto front finale ─────────────────────────────────────────────
    log.info("=" * 60)
    log.info(f"PARETO FRONT — {len(study.best_trials)} trial(s) non dominati")
    log.info(f"{'#':>4}  {'fast_ret':>10}  {'true_rew':>10}  {'succ':>6}  params")

    pareto = sorted(study.best_trials, key=lambda t: t.values[0], reverse=True)
    for t in pareto:
        log.info(
            f"{t.number:>4}  {t.values[0]:>10.3f}  {t.values[1]:>10.4f}  "
            f"{t.values[2]:>6.3f}  {t.params}"
        )

    log.info("=" * 60)
    log.info(f"Risultati salvati in: {output_dir}")
    log.info(f"  pareto_front.json  — Pareto front ordinata per fast_return")
    log.info(f"  all_trials.json    — tutti i trial completati")


if __name__ == "__main__":
    main()