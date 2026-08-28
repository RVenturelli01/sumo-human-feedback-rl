# plots/ — motore di grafici interattivo per la campagna di budget curves

Toolkit per esplorare e graficare le run W&B della tesi. Indicizza tre
progetti insieme — `thesis-grad-diagnostics` (schemi di fusione, ablation della
normalizzazione, frozen probe), `tuning-thesis-budget-curves-completion`
(campagna di budget curves) e `thesis` (ablation a sorgente singola) — perché un
confronto ibrido vs solo-preferenze vs solo-dimostrazioni pesca da progetti
diversi. La colonna `project` è un filtro, quindi si isola una campagna alla
volta con una pillola. Ispirato nell'architettura a un
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
`hybrid_demo_2_soft`, `hybrid_demo_2_bernoulli`.

**`arm` non basta per grad-diagnostics.** Lì i bracci differiscono per *come*
combinano i due gradienti, non per quali sorgenti usano: baseline, prova 1,
prova 2 e le altre varianti hanno tutte `arm = hybrid_demo_2_soft`. Serve la
colonna **`fusion`** (da `algo.kwargs.gcl_fusion`, `norm_balance` quando la
chiave manca perché è il default del costruttore), che sta apposta fuori da
`arm`: `arm` risponde a "quali sorgenti", `fusion` a "come le mette insieme", e
tenerle separate lascia `arm` confrontabile fra i tre progetti. Derivarlo dalla config (non
dal nome del gruppo W&B) è deliberato: durante la campagna sono comparsi nomi
di gruppo non previsti da `scripts/report_budget_curves.py`
(`budget_demo_2_no_norm_*`, `budget_hybrid_demo_2_bern_hom_*`, `..._trmatch_*`)
che quello script colassa silenziosamente nell'arm base — qui restano
classificati correttamente perché si legge cosa la run ha davvero fatto, non
come si chiama.

## Due tipi di grafico, due pipeline dati

### Esclusioni della copertura

Le caselle nella tabella di **Copertura** tolgono singole run dalla figura, e le
esclusioni **sopravvivono al cambio dei filtri**: sono memorizzate per `run_id`,
quindi restano identificabili qualunque filtro si applichi. Le run che entrano
allargando un filtro arrivano selezionate; quelle che avevi tolto restano tolte.
Il badge *«escluse: N»* accanto al conteggio dice quante sono, e il pulsante
*«includi tutte»* le azzera.

### Le formule accanto al grafico

Il selettore mostra a destra dell'anteprima un pannello **Definizioni** con la
formula della metrica scelta e le equazioni degli schemi di fusione presenti
nella selezione (comprese quelle di α e degli update di Adam `u_p`, `u_d`).
Si mostra e si nasconde con la casella **formule** nella barra dell'Anteprima.
La stessa casella decide se l'immagine esportata contenga o meno il pannello:
figura e definizioni finiscono affiancate in un unico file, con i nomi di serie
eventualmente rinominati a mano. L'export `.tex` le lascia sempre fuori — la'
le formule hanno senso come macro, non come immagine.

Le definizioni stanno in **[`rtplots/formulas.py`](rtplots/formulas.py)**,
trascritte da `gradient_statistics.py` e da
`hybrid_algorithm.py:_alpha_weight/_fusion_components`: se una formula e il
codice divergono, è la formula qui a essere sbagliata. Un test verifica che
ogni metrica dei gruppi *Gradienti* e *Normalizzazione* e ogni fusione ancora
implementata abbiano la loro voce.

Il rendering passa da **mathtext di matplotlib**, non da MathJax/KaTeX: la
pagina resta senza dipendenze esterne. Mathtext copre un sottoinsieme di LaTeX
(niente `align`, niente `\text{}`), quindi ogni riga è una formula a sé.

### `--compare-fusion` e `--compare-norm` (solo curve di budget)

Nelle curve di budget la serie di default è **un arm = una curva**, per un motivo
deliberato: a ogni livello di budget gira il best-config *di quel livello*, quindi
quasi ogni iperparametro covaria col livello e `auto_hue` lo terrebbe per una
dimensione vera. Ma `fusion` non è un proxy del livello: senza dirlo, i sei-otto
schemi dello stesso arm finiscono **mediati nella stessa curva**.

Lo stesso vale per `normalize_agent_reward`: ON e OFF dello stesso braccio
finirebbero nella stessa curva.

`--compare-fusion` e `--compare-norm` (checkbox *«confronta gli schemi di
fusione»* e *«confronta normalizzazione on/off»* nel selettore, visibili solo in
modalità budget) aggiungono quelle colonne all'identità della serie. Disattivate
— il default — tutto resta come prima, ma se una delle due sta per essere
mediata viene stampato l'avviso con il flag che la separa.

### Serie che finirebbero identiche

Le regole di `style.toml` applicano la **prima che combacia**, quindi non possono
esprimere «colore dal braccio, tratto dall'ablation»: confrontando due
configurazioni dello stesso braccio (normalizzazione on/off, schemi di fusione,
…) la regola per `arm` le colorerebbe entrambe uguali e le curve sarebbero
indistinguibili.

`figure.py:_decollide` interviene dopo: il **colore** resta quello della regola,
il **tratto** separa le serie che altrimenti coinciderebbero. Vale in tutte le
modalità e per qualunque dimensione, non solo per la normalizzazione — è nato
proprio perché il problema si era già presentato con `fusion`.

Dentro un gruppo che collide, le configurazioni «di base» prendono il tratto
continuo: fra due valori booleani va per prima la condizione **disattivata**,
così l'ablation è sempre la tratteggiata e la convenzione non cambia da una
figura all'altra. (La legenda invece elenca i booleani con `sì` per primo: sono
due ordinamenti diversi, di proposito.)

Le **curve di apprendimento** non hanno bisogno del flag: lì `auto_hue` separa già
per `fusion` da solo.

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
| `fusion` | schema di fusione dei due gradienti (solo hybrid): `norm_balance`, `alpha_norm_single_adam` (prova 1), `dual_adam_alpha` (prova 2), `dual_adam_sum`, `dual_adam_alpha_unit`, `dual_adam_alpha_unit_nobudget`, più due schemi storici poi rimossi |
| `budget_level` | il numero finale del nome gruppo, in entrambe le sintassi: `budget_<...>_<N>` (budget-curves) e `gd_<...>_B<N>` (grad-diagnostics). Robusto per qualunque arm, anche quando query_budget e demo_budget variano insieme |
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

# gli schemi di fusione a confronto, normalizzazione off
.venv/bin/python plots/scripts/plot_budget.py --compare-fusion --name fusioni \
    --filter project=thesis-grad-diagnostics arm_family=hybrid \
             pref_labels=soft normalize_agent_reward=False

# ablation della normalizzazione: una colonna per stato del flag
.venv/bin/python plots/scripts/plot_curves.py --cols normalize_agent_reward \
    --hue fusion --metric reward/grad_probe_dir_var_demo --name ablation_norm \
    --filter project=thesis-grad-diagnostics arm_family=hybrid pref_labels=soft

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
  git (vedi `plots/.gitignore`). **Solo le run `finished` finiscono su disco**:
  la curva di una run ancora in corso e' destinata ad allungarsi, e una cache
  parziale resterebbe li' per sempre accorciando in silenzio ogni figura che la
  usa — `curves.aggregate` fissa la griglia comune sulla run *piu' corta* del
  gruppo. Per ripulire cache scritte prima di questa regola:

  ```bash
  .venv/bin/python plots/scripts/clean_curve_cache.py          # elenca
  .venv/bin/python plots/scripts/clean_curve_cache.py --apply  # cancella
  ```

  Riconosce i file sospetti confrontando ogni run con le sorelle dello stesso
  gruppo sulla stessa metrica, quindi funziona su entrambi gli assi x senza
  doverli distinguere. Quando una serie viene comunque accorciata, l'anteprima
  e la riga di comando lo dicono invece di lasciarti indovinare.
* `RTPLOTS_WANDB_PROJECTS` (lista separata da virgole) sceglie quali progetti
  indicizzare; di default tutti e tre. Aggiungerne uno è una riga in
  `rtplots/paths.py`, non un modulo nuovo: la convenzione di lettura
  (`rtplots/source.py`) è la stessa per ogni progetto generato da
  `experiments/train.py`. Un gruppo con una sintassi di livello nuova
  invece va aggiunto a `parse_group`, altrimenti `budget_level` resta vuoto e
  le curve di budget di quel progetto non escono.
* Le regole di colore in `style.toml` per `fusion` stanno **prima** di quelle
  per `arm`: vince la prima che combacia, e senza quell'ordine gli otto schemi
  uscirebbero tutti del colore di `hybrid_demo_2`.
* Le run di tuning (`tune_*`, livello trial Optuna) restano a
  `scripts/report_tuning.py`: non hanno una nozione di "seed" allo stesso modo
  delle run finali, e la loro analisi (fANOVA, pruning) non si incastra nel
  modello a griglia di questo motore.
* `tikzplotlib` (0.10.1, fermo al 2022) non regge matplotlib 3.6+/numpy
  2/webcolors 24+ senza gli alias applicati in `rtplots/tikz.py:_compat()`:
  vedi `plots/requirements.txt` per le versioni compatibili.
