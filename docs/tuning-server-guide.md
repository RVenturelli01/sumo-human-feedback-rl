# Guida: campagna di tuning e run finali (SAC) sul server

Campagna Optuna per i 6 bracci di `HybridAlgorithm`, seguita dalle curve di
budget e dalle run finali a 5 seed per la tesi:

| Braccio | Significato | Preset |
|---|---|---|
| `pref_soft` | solo preferenze, etichette soft | `demo_weight=0`, `labels_type=soft` |
| `pref_bernoulli` | solo preferenze, etichette binarie campionate | `demo_weight=0`, `labels_type=binary_bernoulli` |
| `demo_1` | solo dimostrazioni, differenza di medie | `total_queries=0`, `loss_type=demo_1` |
| `demo_2` | solo dimostrazioni, surrogato MaxEnt | `total_queries=0`, `loss_type=demo_2` |
| `hybrid_demo_1` | preferenze + dimostrazioni, loss demo_1 | budget pref+demo, `demo_weight` tunato |
| `hybrid_demo_2` | preferenze + dimostrazioni, loss demo_2 | budget pref+demo, `demo_weight` tunato |

Progetti W&B: **`tuning-thesis`** (trial Optuna + curve di budget),
**`thesis`** (run finali a 5 seed, raggruppate per braccio).

Server: `ssh fis3@10.79.4.125`, repo `/work/fis3/sumo-human-feedback-rl/`,
**core 33–47** (15 core = 5 slot × 3 core; ogni run usa `n_envs=2` → 2 processi
SUMO + learner single-thread).

## 0. Prerequisiti (dopo ogni pull)

```bash
cd /work/fis3/sumo-human-feedback-rl
git pull && git -C human-feedback-rl pull   # oppure: git submodule update --remote
pip install -e human-feedback-rl            # il package è cambiato (v0.3.0)
mkdir -p logs outputs/optuna
# se ci sono ancora worker della vecchia campagna:
pkill -f tune_hybrid_sac; pkill -f test_hybrid_SAC
```

## 1. Tuning: tutti i bracci in parallelo, 1 trial alla volta per braccio

Un worker sequenziale per braccio: dopo gli 8 trial casuali di startup, ogni
trial successivo è informato da **tutti** quelli completati (TPE al meglio).

```bash
tmux new -s tuning
./launchers/run_optuna_parallel_arms.sh 30
# = 30 trial per braccio, core da 33, bracci di default:
#   pref_soft pref_bernoulli demo_1 demo_2 hybrid_demo_1  (5 slot)
```

Il sesto braccio non deve aspettare uno slot libero: ogni trial usa in
pratica ~1 core (il collo di bottiglia è il loop di update SAC single-thread,
non SUMO — misurato: pool al 35%, processi env a ~0%), quindi si può
sovrapporre all'intero pool e lo scheduler lo piazza nei cicli liberi:

```bash
nohup python scripts/tune_hybrid_sac.py --arm hybrid_demo_2 --n-trials 30 \
    --cores 33-47 --total-timesteps 1000000 > logs/optuna_hybrid_demo_2.log 2>&1 &
```

Per lo stesso motivo NON conviene alzare `n_envs` (2 basta: lo stepping SUMO
è ~10-30s su ~2-3 min di iterazione) né dare più thread a torch (reti troppo
piccole: +13% misurato, non ripaga un riavvio).

Suggerimento per `hybrid_demo_2` (e volendo anche per rifinire `hybrid_demo_1`):
riparti a caldo dai vincitori delle baseline già completate:

```bash
python scripts/export_best_config.py --arm pref_soft --format params > /tmp/warm.json
# (unisci a mano i param demo del vincitore demo_2 e un demo_weight=1.0, poi)
./launchers/run_optuna_workers.sh hybrid_demo_2 30 1 45 --enqueue-params /tmp/warm.json
```

Parametri tunati per trial: `lr_rew`, `gradient_steps_rew`, `l2_rew`,
`net_arch` reward ([8,8]…[128,128]), `initial_agent_timesteps`; per i bracci
pref anche `batch_size_pref`, `query_schedule`, `initial_queries` (scelte =
2–20% del budget), `fragmenter_type`; per i demo anche `batch_size_expert`,
`batch_size_model`; per gli hybrid anche `demo_weight` (0.1–10 log). Fissi:
oracolo (`pref_temperature=20`, `preference_fragment_length=1`), SAC (con
`gradient_steps=32` = replay ratio storico 2.0 a `n_envs=2`), `n_ensembles=3`.
Budget di tuning: 5000 query / 500 traiettorie, **tranne `pref_bernoulli`:
100000 query** (etichette campionate = rumorose; in reward_label_experiments
a 2M step servivano ≥200K query, riscalate a 100K per 1M — a 5K il braccio
non impara affatto).

### Ri-tunare un braccio con un budget diverso

Mai mescolare budget diversi nello stesso studio (i valori objective non sono
confrontabili e il TPE si corrompe): usa `--study-suffix` per crearne uno
nuovo. Esempio, bernoulli a 100K (dopo aver fermato SOLO il suo worker):

```bash
pkill -f "arm pref_[b]ernoulli"; sleep 2; pkill -f "optuna,pref_[b]ernoulli"
nohup python scripts/tune_hybrid_sac.py --arm pref_bernoulli --n-trials 30 \
    --cores 36-38 --pref-budget 100000 --study-suffix _q100k \
    --total-timesteps 1000000 > logs/optuna_pref_bernoulli_q100k.log 2>&1 &
```

Le run W&B diventano `pref_bernoulli_q100k-t000` (group
`tune_pref_bernoulli_q100k`); per leggere il best trial aggiungi
`--study-suffix _q100k` a `export_best_config.py`, o `STUDY_SUFFIX=_q100k`
per `run_final_5seeds.sh` / `run_budget_curves.sh`.

Durata attesa: ~2.2h/trial a 1M timesteps → 30 trial ≈ 2.5–3 giorni per
braccio, tutti in parallelo (meno con il pruning). Nomi run: `pref_soft-t012`,
group `tune_pref_soft`, tag `[optuna, <arm>]`.

### Monitoraggio

```bash
tail -f logs/optuna_*.log
python scripts/export_best_config.py --arm pref_soft --format summary --top-k 5
```

Stato completo degli studi:

```bash
python - << 'EOF'
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
storage = JournalStorage(JournalFileBackend("outputs/optuna/journal.log"))
for name in optuna.get_all_study_names(storage):
    s = optuna.load_study(study_name=name, storage=storage)
    arm = name.removeprefix("hybrid_sac_")
    states = [t.state.name for t in s.trials]
    done = [t for t in s.trials if t.value is not None]
    line = f"{arm}: {len(s.trials)} trial ({states.count('COMPLETE')} ok, {states.count('PRUNED')} pruned, {states.count('FAIL')} fail)"
    if done:
        best = max(done, key=lambda t: t.value)
        line += f"  best #{best.number}: {best.value:.2f}"
    print(line)
EOF
```

### Interrompere / riprendere

Lo stato vive in `outputs/optuna/journal.log`: `pkill -f tune_hybrid_sac;
pkill -f test_hybrid_SAC`, poi rilancia con i trial residui — gli studi
riprendono da dove erano (`load_if_exists`).

## 2. Curve di budget (dopo il tuning delle 4 baseline)

Griglia 1D per asse (nessun ottimizzatore): livelli × 3 seed con la config
migliore. Per rispettare la scadenza puoi limitarti a `pref_soft` e ai due
bracci demo, o ridurre i livelli con `LEVELS=...`.

```bash
./launchers/run_budget_curves.sh pref_soft        # total_queries: 10000..500
STUDY_SUFFIX=_q100k ./launchers/run_budget_curves.sh pref_bernoulli  # 250000..10000
./launchers/run_budget_curves.sh demo_1           # n_traiettorie: 2723..50
./launchers/run_budget_curves.sh demo_2
```

**Budget minimo (X per le preferenze, Y per le demo)** = il livello più
piccolo con media a 3 seed di `sweep/mean_fast_return` **e**
`sweep/success_rate` ≥ 90% del livello massimo, con anche il livello
successivo che passa. Le run vanno su `tuning-thesis` con group
`budget_<arm>_<livello>` e tag `budget_curve`.

## 3. Run finali a 5 seed (project `thesis`)

Baseline (usa i budget minimi trovati al punto 2, qui a titolo d'esempio
X=2000, Y=200; senza indicazioni usa i default 5000/500):

```bash
PREF_BUDGET=2000 ./launchers/run_final_5seeds.sh pref_soft
PREF_BUDGET=2000 ./launchers/run_final_5seeds.sh pref_bernoulli
DEMO_BUDGET=200  ./launchers/run_final_5seeds.sh demo_1
DEMO_BUDGET=200  ./launchers/run_final_5seeds.sh demo_2
```

Hybrid, due strategie di budget (X, Y = budget delle baseline):

```bash
# Strategia A: metà budget per fonte (X/2 preferenze + Y/2 demo)
PREF_BUDGET=1000 DEMO_BUDGET=100 ./launchers/run_final_5seeds.sh hybrid_demo_1 _A
PREF_BUDGET=1000 DEMO_BUDGET=100 ./launchers/run_final_5seeds.sh hybrid_demo_2 _A
# Strategia B: budget pieni (X preferenze + Y demo)
PREF_BUDGET=2000 DEMO_BUDGET=200 ./launchers/run_final_5seeds.sh hybrid_demo_1 _B
PREF_BUDGET=2000 DEMO_BUDGET=200 ./launchers/run_final_5seeds.sh hybrid_demo_2 _B
```

Ogni chiamata lancia 5 run (una per seed, 3 core l'una, tutte insieme sui
core 33–47) a 2M timesteps, group W&B `<arm><suffisso>`, nomi
`hybrid_demo_1_A-seed3`. La config viene letta dal journal Optuna al volo.

## Note tecniche

- `n_envs=1` non funziona (libsumo = una simulazione per processo); minimo 2.
- I nomi storici delle loss sono cambiati: `demo`→`demo_1`, `maxent_2`→`demo_2`;
  le altre (maxent, corrected) sono state rimosse. `PreferenceAlgorithm` e
  `DemoAlgorithm` non esistono più: tutto passa da `HybridAlgorithm`.
- File per run: `metrics.jsonl` (per il pruner), `final_eval.json` (objective),
  `agent_final.zip`.
- Il journal Optuna non è condivisibile tra macchine: tutto gira sul server.
- `run_optuna_workers.sh <arm> <trial> <n_worker> <primo_core>` resta
  disponibile per mettere più worker su un singolo braccio (trial concorrenti,
  TPE meno informato).
