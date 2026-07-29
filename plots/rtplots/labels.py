"""Nomenclatura: dalle colonne dell'indice alle etichette usate nei grafici.

Molto piu' semplice del progetto di ispirazione (che doveva tradurre famiglia +
IS nei nomi del paper, ωPPO-U/ωPPO-BH): qui il nome della serie e' il nome
dell'arm stesso (`schema.ARM_NAMES`), gia' leggibile.
"""
from __future__ import annotations

from . import rules as R
from . import schema
from .style import mathtt


def arm_name(row, latex: bool = False) -> str:
    """Nome dell'algoritmo (senza iperparametri), gia' pronto da stampare.

    Se una regola `[[series]]` di style.toml copre questa serie, il nome scritto
    li' vince e viene usato **tale e quale**.
    """
    rule = R.rule_for(row)
    chosen = rule.get("latex") if latex else rule.get("name")
    if chosen:
        return str(chosen)
    arm = row.get("arm")
    return schema.ARM_NAMES.get(arm, str(arm) if arm else "?")


def series_label(row, fields=("arm",), paper: bool = True, latex: bool = False) -> str:
    """Etichetta di una serie: nome dell'arm + quello che il resto di `fields`
    aggiunge alla legenda (budget, seed, ...). `paper` non e' usato qui (tenuto
    per compatibilita' di firma con `figure.py`, che lo passa sempre)."""
    fields = set(fields)
    written = (R.rule_for(row) or {}).get("latex" if latex else "name")
    head = str(written) if written else mathtt(arm_name(row))
    extras = []
    for f in schema.FIELDS:
        if f.col not in fields or f.col == "arm":
            continue
        bit = schema.legend_bit(f.col, row.get(f.col))
        if bit:
            extras.append(bit)
    return head + (": " + ", ".join(extras) if extras else "")


def panel_title(field: str, value, paper: bool = True) -> str:
    return schema.panel_title(field, value, paper)
