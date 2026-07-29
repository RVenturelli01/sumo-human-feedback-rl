# plots/ — motore di grafici interattivo per la campagna di budget curves

Toolkit per esplorare e graficare le run W&B della campagna di tuning (progetto
`tuning-thesis-budget-curves-completion`). Ispirato nell'architettura a un
toolkit di plotting per un altro progetto (griglia di pannelli guidata da
filtri, selettore web, export LaTeX), riscritto da zero sullo schema dati di
**questa** repo: bracci (arm), budget di query/traiettorie, curve di
apprendimento e curve di budget.

## Cos'è un "arm" qui

Non esiste un campo "algoritmo" nella config — si deriva da quattro chiavi,
esattamente come fa `scripts/tune_hybrid_sac.py`:

* `algo.kwargs.total_queries > 0` → usa le preferenze
* `algo.kwargs.demo_weight > 0` → usa le dimostrazioni
* entrambe vere → **hybrid**, solo la prima → **pref**, solo la seconda → **demo**

Otto valori possibili di `arm`: `demo_1`, `demo_2`, `pref_soft`,
`pref_bernoulli`, `hybrid_demo_1_soft`, `hybrid_demo_1_bernoulli`,
`hybrid_demo_2_soft`, `hybrid_demo_2_bernoulli`. Derivarlo dalla config (non
dal nome del gruppo W&B) è deliberato: durante la campagna sono comparsi nomi
di gruppo non previsti da `scripts/report_budget_curves.py`
(`budget_demo_2_no_norm_*`, `budget_hybrid_demo_2_bern_hom_*`, `..._trmatch_*`)
che quello script colassa silenziosamente nell'arm base — qui restano
classificati correttamente perché si legge cosa la run ha davvero fatto, non
come si chiama.

## Due tipi di grafico, due pipeline dati

| | curva di apprendimento | curva di budget |
|---|---|---|
| script | `plot_curves.py` | `plot_budget.py` |
| un run vale | una serie storica (return vs timestep) | un solo numero (eval finale) |
| fonte | history W&B (`agent/rewards/ep_fast_return`, ...) | `run.summary` (`sweep/*`) |
| asse x | tempo (timestep o iterazione) | livello di budget, **scala log** |
| aggregazione | media + banda sui seed, interpolata su una griglia comune | media + errorbar sui seed, un punto per livello |
| modulo | `rtplots/curves.py` | `rtplots/budget.py` |

Tutto qui passa da W&B: niente file locali raggiungibili (`metrics.jsonl`,
`evaluations.npz` vivono sul server dove giri il training, non sulla macchina
da cui analizzi, vedi `docs/analysis-pipeline-guide.md`). Ogni run non ancora
in cache costa una richiesta di rete; i risultati restano in
`plots/.cache/` (curve) e in cache per run (summary), quindi solo la prima
volta è lenta.

## Struttura

```
plots/
├── rtplots/             libreria
│   ├── schema.py        i campi dell'indice: titoli, filtri, ruolo nelle figure
│   ├── source.py        una run W&B -> riga dell'indice (deriva l'arm dalla config)
│   ├── index.py         indice dei run: metadati -> parquet in cache
│   ├── metrics.py        catalogo delle metriche (curva vs eval finale, con l'asse x giusto)
│   ├── curves.py         curve di apprendimento (history W&B) + aggregazione sui seed
│   ├── budget.py         eval finale (summary) aggregata per livello di budget + regola del 90%
│   ├── select.py         filtri e conteggi di copertura
│   ├── figure.py         FigureSpec + pipeline unica: selezione -> figura (curve o budget)
│   ├── grid.py            disegno della griglia di pannelli (banda o errorbar, lineare o log-x)
│   ├── tikz.py            export .tex (pgfplots), un file per pannello
│   ├── selection.py       selezioni salvate (lettura, scrittura)
│   ├── labels.py          nomi delle serie (nome dell'arm + budget/seed in legenda)
│   ├── rules.py           legge style.toml (le regole scritte a mano)
│   ├── style.py           traduce le regole in rcParams e colori
│   └── webui/            selettore: api.py (logica) + server.py (HTTP)
├── scripts/              eseguibili
│   ├── build_index.py    costruisce/aggiorna la cache dei metadati
│   ├── selector.py       selettore interattivo delle run (server locale)
│   ├── list_runs.py      che run/seed esistono per una data combinazione
│   ├── plot_curves.py    curve di apprendimento (metrica vs timestep/iterazione)
│   ├── plot_budget.py    curve di budget (eval finale vs livello di budget)
│   └── prefetch_curves.py  scarica in blocco curve/summary prima di aprire il selettore
├── style.toml            regole dei grafici: palette per arm, si modifica a mano
├── selector/             pagina del selettore (html/css/js, nessuna dipendenza)
├── tests/                test senza rete (`.venv/bin/python -m pytest plots/tests -q`)
├── requirements.txt      dipendenze in piu' rispetto al resto della repo
└── output/               figure generate (ignorata da git)
```

**Una sola pipeline.** Riga di comando e selettore costruiscono lo stesso
oggetto — `FigureSpec` (con `kind="curve"` o `kind="budget"`) — e lo passano a
`rtplots.figure`: la stessa selezione dà la stessa figura da tutte e due le
strade.

## Uso rapido

```bash
# prima volta (o dopo una nuova campata di run)
.venv/bin/python plots/scripts/build_index.py

# selettore interattivo
.venv/bin/python plots/scripts/selector.py     # http://127.0.0.1:8770
```

Filtri per ogni dimensione (algoritmo, budget, ...), un pulsante per il tipo di
grafico (curva di apprendimento / curva di budget), tendina "cosa plottare"
(diversa nei due modi: la prima elenca le metriche-curva, la seconda le
`sweep/*`), righe/colonne/colori, anteprima in tempo reale, tabella di
copertura (quanti seed per combinazione), selezioni salvabili con un nome,
export JPEG o LaTeX (`.tex` pgfplots, un file per pannello + lo snippet
`\begin{figure}` già montato).

## Filtri disponibili

Ogni script accetta `--filter chiave=valore …` (in AND), stessa sintassi delle
pillole del selettore (`arm!=demo_1,demo_2`, `query_budget>=5000`, ...):

| colonna | cosa filtra |
|---|---|
| `arm` | i 4 bracci baseline + le 4 combinazioni di hybrid (demo_loss × pref_labels) |
| `arm_family` | `demo` \| `pref` \| `hybrid` |
| `demo_loss` | `demo_1` \| `demo_2` (per demo-only e hybrid) |
| `pref_labels` | `soft` \| `bernoulli` (per pref-only e hybrid) |
| `demo_mode` | `gcl` \| `preferences` (baseline Ibarz, se/quando comparirà) |
| `query_budget` | `algo.kwargs.total_queries` |
| `demo_budget` | `run.n_expert_trajectories` (mancante = dataset intero, non un valore ignoto) |
| `budget_level` | il numero finale del nome gruppo (`budget_<...>_<N>`): robusto per qualunque arm, anche quando query_budget e demo_budget variano insieme (i bracci hybrid budget-*, dove il livello è la somma dei due) |
| `normalize_agent_reward`, `initial_queries`, `demo_weight`, `query_schedule`, `fragmenter_type`, `pref_temperature`, `reward_net_arch`, `demo_subsample_seed`, `total_timesteps` | iperparametri della run |
| `state`, `project`, `group_tag` | stato W&B, progetto, e l'etichetta libera fra `budget_` e il livello (distingue varianti come `_no_norm`, `_bern_hom`, `_soft_trmatch` non ancora modellate come colonne proprie) |

Per default vengono usati solo i run `finished` (`--state any` per includere
gli altri).

## Esempi

```bash
# tutti gli arm, una serie ciascuno, curva di apprendimento
.venv/bin/python plots/scripts/plot_curves.py --name learning_all

# solo le combinazioni hybrid, una colonna per tipo di etichette
.venv/bin/python plots/scripts/plot_curves.py --filter arm_family=hybrid \
    --cols pref_labels --hue demo_loss --name hybrid_learning

# curve di budget dei bracci pref (query, return finale) + regola del 90%
.venv/bin/python plots/scripts/plot_budget.py --filter arm_family=pref --name budget_pref

# curve di budget dei demo-only sul success rate, asse x = traiettorie
.venv/bin/python plots/scripts/plot_budget.py --filter arm_family=demo \
    --metric sweep/success_rate --budget-x demo_budget --name budget_demo_success

# cosa c'è a disposizione
.venv/bin/python plots/scripts/list_runs.py --by arm
.venv/bin/python plots/scripts/list_runs.py --filter arm_family=hybrid --by arm demo_loss pref_labels
```

`--list-metrics` su entrambi gli script stampa il catalogo delle metriche
disponibili (qualsiasi altra chiave loggata su W&B è comunque accettata: l'asse
x si indovina dal prefisso, `agent/*` → timestep, il resto → iterazione).

## Stile

Le regole stanno in **[`style.toml`](style.toml)**: palette per arm (stessa di
`scripts/_report_common.py:ARM_COLORS`, così un braccio ha sempre lo stesso
colore in tutte le figure della tesi), spessori, banda, legenda, nomi delle
serie e macro LaTeX. Le otto combinazioni di hybrid condividono il colore del
loro `demo_loss` e si distinguono per tratteggio (soft = continuo, bernoulli =
tratteggiato). Vale sia per l'anteprima sia per l'export `.tex`, riletto a ogni
figura (si salva e si ridisegna senza riavviare il selettore).

## Note

* Cache in `plots/.cache/` (override con `RTPLOTS_CACHE`), mai nella repo su
  git (vedi `plots/.gitignore`).
* `RTPLOTS_WANDB_PROJECTS` (lista separata da virgole) sceglie quali progetti
  indicizzare; di default solo `tuning-thesis-budget-curves-completion`. Se la
  campagna si allarga ad altri progetti (es. una run finale a 5 seed in un
  progetto `thesis` separato) è una riga in più in `rtplots/paths.py`, non un
  modulo nuovo: la convenzione di lettura (`rtplots/source.py`) è la stessa per
  ogni progetto generato da `scripts/train_hybrid_sac.py`.
* Le run di tuning (`tune_*`, livello trial Optuna) restano a
  `scripts/report_tuning.py`: non hanno una nozione di "seed" allo stesso modo
  delle run finali, e la loro analisi (fANOVA, pruning) non si incastra nel
  modello a griglia di questo motore.
* `tikzplotlib` (0.10.1, fermo al 2022) non regge matplotlib 3.6+/numpy
  2/webcolors 24+ senza gli alias applicati in `rtplots/tikz.py:_compat()`:
  vedi `plots/requirements.txt` per le versioni compatibili.
