# Tuning diagnostics dashboard

Dashboard locale per analizzare i tuning della tesi. Legge score e metadati dal
progetto W&B `tuning-thesis` e li unisce ai parametri dei trial nel journal
Optuna. Non modifica run, study o file degli esperimenti.

## Avvio

Sul server:

```bash
cd /work/fis3/sumo-human-feedback-rl/dashboard/tuning
.venv/bin/python app.py --sync-only
.venv/bin/python app.py
```

Sul computer locale, in un altro terminale:

```bash
ssh -L 8050:127.0.0.1:8050 fis3@10.79.4.125
```

Aprire `http://127.0.0.1:8050`.

## Fonti dati

- W&B: stato, score `sweep/mean_fast_return`, runtime e identificativo run.
- Optuna: stato del trial e iperparametri, letti in sola lettura da
  `/work/fis3/sumo-human-feedback-rl/outputs/optuna/journal.log`.
- Cache: `cache/tuning_runs.json`, sostituita atomicamente a ogni sync.

Il pulsante `Sincronizza W&B` aggiorna la cache. In caso di errore mantiene
l'ultimo snapshot valido.

## Diagnostiche

- Evoluzione dello score e best-so-far.
- Gap tra i primi candidati e densita vicino al massimo.
- Distanza Gower-like tra configurazioni, con normalizzazione min-max e scala
  log per parametri positivi che coprono almeno due ordini di grandezza.
- Associazione univariata: correlazione di Spearman per parametri numerici ed
  eta-quadro per parametri categorici. Non e una misura causale.
- Parameter importance fANOVA di Optuna, calcolata sui soli trial completati,
  normalizzata a somma 1 e resa riproducibile con seed fisso `0`. Il selettore
  nell'interfaccia permette di confrontarla con l'associazione univariata.
- Coordinate parallele e tabella filtrabile dei top-k.

Le etichette diagnostiche sono euristiche descrittive. Non sostituiscono una
rivalutazione multi-seed dei candidati.

## Esportazione

La toolbar di ogni grafico contiene il pulsante Plotly per scaricare un PNG a
risoluzione 3600 x 2000. Prima dell'esportazione, impostare algoritmo, top-k e
parametro desiderati.

## Installazione isolata

La `.venv` vive dentro questa directory ed eredita i pacchetti scientifici
dall'ambiente `sumo-rlhf`. Le sole dipendenze aggiunte sono elencate in
`requirements.txt`.
