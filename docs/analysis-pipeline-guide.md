# Guida: pipeline di analisi dei risultati

Tre script in `scripts/` trasformano i dati W&B in figure (PNG + PDF) e
tabelle (Markdown + LaTeX) pronte per la tesi. Tutti scrivono in
`reports/thesis/` (cartella creata al primo uso, override con `--out`),
leggono W&B via API (serve `wandb login` sulla macchina dove li lanci — vanno
benissimo dal Mac, non serve il server) e sono **rilanciabili in ogni
momento**: sovrascrivono i propri output, quindi si possono usare sui dati
parziali oggi e rilanciare identici a campagna finita.

I colori seguono il braccio in ogni figura (palette validata per daltonismo):
pref_soft=blu, pref_bernoulli=verde acqua, demo_1=giallo, demo_2=verde,
hybrid_demo_1=viola, hybrid_demo_2=rosso. Le varianti dello stesso braccio
(`_q100k`, strategie `_A`/`_B`) tengono il colore e cambiano tratteggio.

## 1. `report_tuning.py` — com'è andata la ricerca Optuna

```bash
python scripts/report_tuning.py
# con le importanze dei parametri (serve una COPIA del journal, mai il file vivo):
scp fis3@10.79.4.125:/work/fis3/sumo-human-feedback-rl/outputs/optuna/journal.log /tmp/journal.log
python scripts/report_tuning.py --journal /tmp/journal.log --top-k 5
```

Legge il project `tuning-thesis` (tutti i gruppi `tune_*`, inclusi i suffissi
tipo `tune_pref_bernoulli_q100k`) e produce:

| Output | Contenuto |
|---|---|
| `tuning_top_<braccio>.md/.tex` | top-k trial per braccio: parametri, objective, success/collision, durata |
| `tuning_progress.png/.pdf` | objective vs numero di trial per braccio (punteggiato: best-so-far) |
| `tuning_param_importances.md/.tex` | importanza dei parametri per studio (fANOVA; solo con `--journal`, ≥5 trial completati) |

Note: le importanze richiedono `scikit-learn` (`pip install scikit-learn`,
già installato in `sumo-rlhf` sul Mac). Il journal va **copiato** (scp) e
letto dalla copia: leggere è innocuo, ma non serve rischiare lock sul file su
cui scrivono i worker.

## 2. `report_thesis_runs.py` — la tabella principale e le curve

```bash
python scripts/report_thesis_runs.py                  # project thesis (run finali 5 seed)
python scripts/report_thesis_runs.py --project smoke  # prova dello schema su run di smoke
python scripts/report_thesis_runs.py --include-running  # includi run non ancora finite
```

Raggruppa per `run.group` (`pref_soft`, …, `hybrid_demo_1_A`,
`hybrid_demo_1_B`) e produce:

| Output | Contenuto |
|---|---|
| `thesis_main_table.md/.tex` | **la tabella della tesi**: media ± std sui seed di return, success/collision/off-road/timeout rate, velocità, durata episodi |
| `thesis_main_table_raw.csv` | stessi numeri in forma numerica (mean/std separati) per elaborazioni tue |
| `thesis_learning_curves.png/.pdf` | ep_fast_return vs timesteps, media sui seed con banda ±1 std, una linea per gruppo |

## 3. `report_budget_curves.py` — budget minimo X e Y

```bash
python scripts/report_budget_curves.py     # project tuning-thesis, gruppi budget_*
```

Aggrega i gruppi `budget_<braccio>_<livello>` e produce:

| Output | Contenuto |
|---|---|
| `budget_curve_<braccio>.md/.tex` | per livello: media ± std di return e success rate, n seed |
| `budget_curves.png/.pdf` | due pannelli (return e success — mai doppio asse), x logaritmica, barre d'errore sui seed |
| stdout | **il budget minimo per braccio** con la regola del 90% (entrambe le metriche ≥90% del livello massimo, col livello successivo che passa) |

I budget minimi stampati per i bracci pref (X) e demo (Y) sono i valori da
usare nelle strategie A (`X/2`, `Y/2`) e B (`X`, `Y`) di `run_final_5seeds.sh`.

## Flusso consigliato a fine campagna

```bash
# 1. il tuning è finito: guarda la ricerca e scegli/esporta le best config
python scripts/report_tuning.py --journal /tmp/journal.log
# 2. lancia curve di budget sul server (guida tuning-server-guide.md), poi:
python scripts/report_budget_curves.py        # -> ti dà X e Y
# 3. lancia le run finali 5 seed sul server con X e Y, poi:
python scripts/report_thesis_runs.py          # -> tabella + curve della tesi
```

Ogni script stampa a video un riepilogo (best per braccio, budget minimi,
tabella) oltre a scrivere i file, quindi si può usare anche solo come
monitoraggio veloce durante la campagna.
