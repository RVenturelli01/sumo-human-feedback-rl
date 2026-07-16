# Guida: tuning delle baseline hybrid (SAC) sul server

Campagna Optuna per trovare le migliori configurazioni dei 4 bracci baseline
di `HybridAlgorithm`, minimizzando i campioni di preferenza/dimostrazione:

| Braccio | Significato | Preset |
|---|---|---|
| `pref_soft` | solo preferenze, etichette soft | `demo_weight=0`, `labels_type=soft` |
| `pref_bernoulli` | solo preferenze, etichette binarie campionate | `demo_weight=0`, `labels_type=binary_bernoulli` |
| `demo_demo` | solo dimostrazioni, loss difference-of-means | `total_queries=0`, `loss_type=demo` |
| `demo_maxent2` | solo dimostrazioni, loss MaxEnt-2 | `total_queries=0`, `loss_type=maxent_2` |

Tutto gira sul server (`ssh fis3@10.79.4.125`, repo in
`/work/fis3/sumo-human-feedback-rl/`), **core consentiti 33–47** (15 core →
5 worker × 3 core). Ogni trial è una run `test_hybrid_SAC.py` con `n_envs=2`
(2 processi SUMO + learner single-thread = 3 core).

## 0. Prerequisiti (una tantum, sul server)

```bash
cd /work/fis3/sumo-human-feedback-rl
git pull                          # e git pull nei submoduli se necessario
pip install optuna                # nell'env sumo-rlhf
python -c "import optuna, libsumo; print('ok')"
ls datasets/expert_trajectories_no_collision.pkl   # altrimenti: python scripts/download_datasets.py
wandb login --verify              # deve già essere loggato
mkdir -p logs outputs/optuna
```

## 1. Scelta dell'orizzonte (Fase B)

```bash
python scripts/find_horizon.py
```

Analizza le run W&B storiche pref_only/demo_only e stampa `T_final` e
`T_tune = T_final/2`. Se non trova storia usabile: `T_final=2M`, `T_tune=1M`
(default del tuner). Usa il valore stampato come `--total-timesteps` sotto.

## 2. Ricerca Optuna (Fase C) — un braccio alla volta

Dentro `tmux` (una sessione per braccio, in sequenza):

```bash
tmux new -s tuning
./launchers/run_optuna_workers.sh pref_soft      40 5 33 --total-timesteps 1000000
# quando finisce:
./launchers/run_optuna_workers.sh pref_bernoulli 40 5 33 --total-timesteps 1000000
./launchers/run_optuna_workers.sh demo_demo      40 5 33 --total-timesteps 1000000
./launchers/run_optuna_workers.sh demo_maxent2   40 5 33 --total-timesteps 1000000
```

Argomenti: `<arm> <n_trial_totali> [n_worker=5] [primo_core=33] [extra...]`.
I 40 trial sono divisi tra i 5 worker; ogni worker pinna il suo trial su 3
core (33-35, 36-38, 39-41, 42-44, 45-47). Budget di campioni durante la
ricerca: `--pref-budget 5000` query per i bracci pref, `--demo-budget 500`
traiettorie per i bracci demo (default del tuner).

Cosa fa ogni trial: campiona gli iperparametri (lr_rew, gradient_steps_rew,
batch size, query schedule / net_arch, initial_agent_timesteps), lancia il
training, viene **potato** (MedianPruner) se `rollout/mean_true_reward` è
sotto la mediana dopo il 40% delle iterazioni, e a fine run l'objective è
`eval/mean_fast_return` da 20 episodi deterministici held-out.

### Monitoraggio

```bash
tail -f logs/optuna_pref_soft_w*.log        # log dei worker
ls outputs/optuna/hybrid_sac_pref_soft/     # un dir per trial (train.log, command.txt)
```

Ogni trial è anche una run W&B taggata `[optuna, <arm>]` nel progetto
`preference+demonstration`. Stato dello studio:

```bash
python - << 'EOF'
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
storage = JournalStorage(JournalFileBackend("outputs/optuna/journal.log"))
for arm in ("pref_soft", "pref_bernoulli", "demo_demo", "demo_maxent2"):
    try:
        s = optuna.load_study(study_name=f"hybrid_sac_{arm}", storage=storage)
    except KeyError:
        continue
    df = s.trials_dataframe()
    print(f"\n== {arm}: {len(df)} trial ==")
    print(df[["number", "state", "value"]].tail(10).to_string(index=False))
    done = [t for t in s.trials if t.value is not None]
    if done:
        print("best:", s.best_trial.number, s.best_trial.value, s.best_trial.params)
EOF
```

### Interrompere / riprendere

Lo stato vive in `outputs/optuna/journal.log`: si può uccidere tutto
(`pkill -f tune_hybrid_sac`) e rilanciare `run_optuna_workers.sh` con i trial
residui — lo studio riparte da dov'era (`load_if_exists`). Il trial in corso
al momento del kill risulta FAIL e viene semplicemente rimpiazzato.

## 3. Validazione multi-seed dei top-3 (Fase D)

Per ogni braccio, prendi i 3 trial migliori (dal riepilogo sopra, scartando
config quasi identiche) e rilancia ognuno su seed 1,2,3 a T_final con il
launcher, passando i parametri del trial come override extra. Esempio per un
trial `pref_soft` con `lr_rew=0.0004, gradient_steps_rew=150,
batch_size_pref=128, query_schedule=hyperbolic, initial_queries=250,
initial_agent_timesteps=20000`:

```bash
for SEED in 1 2 3; do
  MODE=pref_only LABELS_TYPE=soft SEED=$SEED N_ENVS=2 \
  TOTAL_TIMESTEPS=2000000 TOTAL_QUERIES=5000 INITIAL_QUERIES=250 \
  QUERY_SCHEDULE=hyperbolic PREF_BATCH_SIZE=128 REWARD_LR=0.0004 \
  REWARD_GRADIENT_STEPS=150 INITIAL_AGENT_TIMESTEPS=20000 \
  WANDB_TAGS="[topk_validation,pref_soft]" \
  taskset -c 33-35 ./launchers/run_hybrid_SAC.sh &
done
```

(Per i bracci demo: `MODE=demo_only LOSS_TYPE=... EXPERT_BATCH_SIZE=...
MODEL_BATCH_SIZE=... REWARD_ARCH=...` e aggiungi in coda l'override
`run.n_expert_trajectories=500`.) Con 3 core per run, sui core 33–47 girano 5
run in parallelo. Selezione: media a 3 seed di `sweep/mean_fast_return` più
alta; a parità entro 1 std, std minore, poi collision rate minore.

## 4. Curve di budget (Fase E)

Con la config vincitrice di ogni braccio, 3 seed × livelli di budget a
T_final, tag `budget_curve`:

- bracci pref: `TOTAL_QUERIES ∈ {10000, 5000, 2000, 1000, 500}` con
  `INITIAL_QUERIES = min(valore tunato, TOTAL_QUERIES/5)`;
- bracci demo: `run.n_expert_trajectories ∈ {2723, 1000, 500, 200, 100, 50}`
  (il sottoinsieme varia col seed: di default segue `run.seed`).

**Budget minimo** = il livello più piccolo la cui media a 3 seed di
`sweep/mean_fast_return` **e** `sweep/success_rate` è ≥ 90% del livello
full-budget, con anche il livello immediatamente superiore che passa.

## Note tecniche

- `n_envs=1` non funziona: DummyVecEnv + 2 vec-env nello stesso processo
  rompono libsumo (una simulazione per processo). Minimo `n_envs=2`.
- `l2_rew` resta a `1e-4`: `1e-2` collassa la reward net (diagnosi 2026-07-05).
- `pref_temperature=20` e `preference_fragment_length=1` definiscono
  l'oracolo sintetico (il problema), non si tunano.
- I file per-run: `metrics.jsonl` (metriche per iterazione, usato dal pruner),
  `final_eval.json` (objective), `agent_final.zip` dentro
  `outputs/optuna/hybrid_sac_<arm>/trial_NNNN/<run name>/`.
- Il journal Optuna non è condivisibile tra macchine (serve un filesystem
  comune): i worker girano solo sul server.
