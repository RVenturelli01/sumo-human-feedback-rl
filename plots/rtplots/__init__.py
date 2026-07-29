"""Utility per i grafici delle run di questa campagna (tuning-thesis-budget-curves-completion).

  schema     i campi dell'indice: titoli, formattazione, ruolo nelle figure
  source     come si legge una run (config Hydra -> riga dell'indice)
  index      metadati delle run (una riga per run), in cache in plots/.cache/
  curves     curve di apprendimento (history W&B) e aggregazione sui seed
  budget     eval finale (run.summary) aggregata per livello di budget
  select     filtri in stile riga di comando
  figure     dalla selezione alla figura: pipeline unica di CLI e selettore
  selection  selezioni salvate dal selettore (lettura, scrittura)
  labels     nomi degli arm e delle serie
  style      palette per arm e look dei grafici
  grid       disegno della griglia di pannelli
  webui      il selettore interattivo
"""
from . import (budget, curves, figure, index, labels, paths, schema, select,
               selection, source, style)  # noqa: F401

__all__ = ["budget", "curves", "figure", "index", "labels", "paths", "schema",
           "select", "selection", "source", "style"]
