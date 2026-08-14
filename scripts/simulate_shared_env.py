#!/usr/bin/env python
"""Simulazione strumentata di una run intera con ambiente CONDIVISO.

Non cerca crash: cerca errori di semantica. Gira il vero
``HybridAlgorithm.train()`` su un ambiente finto e deterministico, registra
cosa accade a ogni passo del ciclo, e verifica invarianti che devono valere
per costruzione. Dove un'invariante non e' verificabile ma la quantita' e'
comunque interessante (per esempio quanto e' "vecchio" il pool di rollout),
il numero viene riportato.

Uso:
    python scripts/simulate_shared_env.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "human-feedback-rl"))
sys.path.insert(0, str(REPO / "human-feedback-rl" / "tests"))

import numpy as np
import torch as th
from stable_baselines3 import SAC

from conftest import FakeVecEnv, make_trajectories
from human_feedback_rl.algorithms import HybridAlgorithm
from human_feedback_rl.common.replay_buffers import RewardRelabelReplayBuffer

EPISODE_LEN = 10
PER_ITER = 100          # timesteps_per_iteration
N_ITER = 6
BOOT = 50               # initial_agent_timesteps


class CountingEnv(FakeVecEnv):
    """Conta i passi realmente eseguiti e marca quando avvengono."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.total_steps = 0
        self.phase = "init"
        self.per_phase: dict[str, int] = {}

    def step_wait(self):
        self.total_steps += self.num_envs
        self.per_phase[self.phase] = self.per_phase.get(self.phase, 0) + self.num_envs
        return super().step_wait()


def build():
    env = CountingEnv(num_envs=1, episode_len=EPISODE_LEN)
    agent = SAC(
        "MlpPolicy", env, buffer_size=5000, learning_starts=0, batch_size=16,
        train_freq=4, gradient_steps=8, policy_kwargs=dict(net_arch=[16]),
        replay_buffer_class=RewardRelabelReplayBuffer, seed=0, verbose=0,
    )
    rng = np.random.default_rng(0)
    algo = HybridAlgorithm(
        env, agent,
        expert_trajectories=make_trajectories(rng, [10] * 6),
        loss_type="demo_2",
        demo_mode="gcl",
        gcl_fusion="alpha_norm_single_adam",
        gradient_steps_rew=3,
        batch_size_expert=4, batch_size_model=4, batch_size_pref=8,
        total_queries=12,
        initial_queries=2,
        preference_fragment_length=1,
        query_schedule="constant",
        relabel_rewards=True,
        normalize_agent_reward=False,
        initial_agent_timesteps=BOOT,
        reward_model_kwargs=dict(n_ensembles=1, net_arch=[8]),
        rng=np.random.default_rng(0),
        output_formats=[],
        rollout_env=None,            # <- AMBIENTE CONDIVISO (default)
    )
    return algo, env


class Trace:
    def __init__(self):
        self.events: list[tuple] = []
        self.pools: list[list] = []          # traiettorie restituite per iterazione
        self.pool_ids: list[set] = []
        self.alpha_calls: list[int] = []     # iterazione in cui alpha e' stimato
        self.pref_counts: list[int] = []     # confronti totali dopo ogni raccolta
        self.rm_before: list = []
        self.rm_after: list = []
        self.steps_at: dict[str, list] = {}


def instrument(algo, env, tr: Trace):
    gen = algo.trajectory_generator

    def snapshot():
        return th.cat([p.detach().reshape(-1).clone()
                       for m in algo.reward_model.members for p in m.parameters()])

    orig_sample = algo.sample_rollout
    orig_feedback = algo._collect_feedback
    orig_train_rm = algo._train_reward_model
    orig_train_agent = algo.train_agent
    orig_alpha = algo._estimate_alpha

    def sample_rollout(*a, **k):
        env.phase = "sample_rollout"
        before = env.total_steps
        out = orig_sample(*a, **k)
        tr.events.append(("sample_rollout", env.total_steps - before, len(out)))
        tr.pools.append(out)
        tr.pool_ids.append({id(t) for t in out})
        env.phase = "altro"
        return out

    def collect_feedback(n, *a, **k):
        env.phase = "feedback"
        out = orig_feedback(n, *a, **k)
        tr.pref_counts.append(len(algo.dataset_train))
        tr.events.append(("feedback", None, f"{n} query, dataset={len(algo.dataset_train)}"))
        env.phase = "altro"
        return out

    def train_rm(*a, **k):
        env.phase = "train_reward"
        before = env.total_steps
        out = orig_train_rm(*a, **k)
        tr.events.append(("train_reward", env.total_steps - before, None))
        env.phase = "altro"
        return out

    def train_agent(steps, *a, **k):
        env.phase = "train_agent"
        before = env.total_steps
        tr.rm_before.append(snapshot())
        out = orig_train_agent(steps, *a, **k)
        tr.rm_after.append(snapshot())
        tr.events.append(("train_agent", env.total_steps - before, steps))
        env.phase = "altro"
        return out

    def estimate_alpha(*a, **k):
        tr.alpha_calls.append(len(tr.events))
        tr.events.append(("estimate_alpha", None, None))
        return orig_alpha(*a, **k)

    algo.sample_rollout = sample_rollout
    algo._collect_feedback = collect_feedback
    algo._train_reward_model = train_rm
    algo.train_agent = train_agent
    algo._estimate_alpha = estimate_alpha
    return gen


def main() -> int:
    algo, env = build()
    tr = Trace()
    gen = instrument(algo, env, tr)

    expert_fp_prima = [len(t) for t in algo.expert_trajectories]
    algo.train(total_timesteps=PER_ITER * N_ITER, timesteps_per_iteration=PER_ITER,
               log_interval=1000)
    expert_fp_dopo = [len(t) for t in algo.expert_trajectories]

    print("=" * 74)
    print("TRACCIA DEL CICLO  (passi ambiente fra parentesi)")
    print("=" * 74)
    for i, (nome, a, b) in enumerate(tr.events):
        extra = "" if a is None else f"  passi={a}"
        extra2 = "" if b is None else f"  -> {b}"
        print(f"  {i:3d}  {nome:16s}{extra}{extra2}")

    print()
    print("=" * 74)
    print("INVARIANTI")
    print("=" * 74)
    esiti = []

    def check(nome, ok, dettaglio=""):
        esiti.append(ok)
        print(f"  [{'OK ' if ok else 'KO '}] {nome}" + (f"   {dettaglio}" if dettaglio else ""))

    # I1 - nessuna transizione persa
    raccolte = sum(len(t) for pool in tr.pools for t in pool)
    aperte = sum(len(t) for t in gen.buffering_wrapper._partial_trajectories)
    non_lette = sum(len(t) for t in gen.buffering_wrapper._finished_trajectories)
    check("ogni passo dell'ambiente e' in un pool, o aperto, o non ancora letto",
          raccolte + aperte + non_lette == env.total_steps,
          f"{raccolte} + {aperte} + {non_lette} = {raccolte+aperte+non_lette} "
          f"contro {env.total_steps}")

    # I2 - nessuna traiettoria usata due volte
    doppioni = 0
    visti = set()
    for ids in tr.pool_ids:
        doppioni += len(visti & ids)
        visti |= ids
    check("nessuna traiettoria compare in due iterazioni", doppioni == 0,
          f"doppioni={doppioni}")

    # I3 - alpha una volta per iterazione, prima dei passi di gradiente
    n_train_rm = sum(1 for e in tr.events if e[0] == "train_reward")
    check("alpha stimato una volta per allenamento del reward model",
          len(tr.alpha_calls) == n_train_rm,
          f"alpha={len(tr.alpha_calls)}  train_reward={n_train_rm}")

    # I4 - reward model congelato durante il training dell'agente
    congelato = all(th.equal(a, b) for a, b in zip(tr.rm_before, tr.rm_after))
    check("il reward model non cambia durante train_agent", congelato)

    # I5 - le dimostrazioni non cambiano
    check("le dimostrazioni restano le stesse per tutta la run",
          expert_fp_prima == expert_fp_dopo)

    # I6 - budget di confronti rispettato
    check("i confronti raccolti non superano total_queries",
          len(algo.dataset_train) <= algo.total_queries,
          f"{len(algo.dataset_train)} su {algo.total_queries}")

    print()
    print("=" * 74)
    print("QUANTITA' DA GUARDARE (non sono invarianti)")
    print("=" * 74)
    print(f"  passi ambiente totali            {env.total_steps}")
    print(f"  dichiarati (total_timesteps)     {PER_ITER * N_ITER}")
    print(f"  ripartizione per fase            {env.per_phase}")
    print(f"  confronti dopo ogni raccolta     {tr.pref_counts}")
    print(f"  traiettorie per pool             {[len(p) for p in tr.pools]}")
    print(f"  transizioni per pool             {[sum(len(t) for t in p) for p in tr.pools]}")

    print()
    return 0 if all(esiti) else 1


if __name__ == "__main__":
    sys.exit(main())
