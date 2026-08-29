"""Figures from the Weights & Biases runs.

  schema     the index fields: titles, formatting, role in a figure
  source     how one run is read (Hydra config -> one index row)
  index      run metadata, one row per run, cached under plots/.cache/
  curves     learning curves from the W&B history, aggregated over seeds
  budget     final evaluation from run.summary, aggregated per budget
  select     command-line style filters
  figure     from a selection to a figure, shared by the CLI and the selector
  selection  selections saved by the selector
  labels     names for the methods and the series
  style      colours per method, and the look of the plots
  grid       drawing the panel grid
  webui      the interactive selector
"""
from . import (budget, curves, figure, index, labels, paths, schema, select,
               selection, source, style)  # noqa: F401

__all__ = ["budget", "curves", "figure", "index", "labels", "paths", "schema",
           "select", "selection", "source", "style"]
