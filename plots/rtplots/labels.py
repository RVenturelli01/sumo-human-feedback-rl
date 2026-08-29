"""From index columns to the labels shown in the figures.

A series is named after its method, which is already readable, so the mapping
is just ``schema.ARM_NAMES``.
"""
from __future__ import annotations

from . import rules as R
from . import schema
from .style import mathtt


def arm_name(row, latex: bool = False) -> str:
    """The method name, ready to print, without hyperparameters.

    A ``[[series]]`` rule in style.toml wins, and is used exactly as written.
    """
    rule = R.rule_for(row)
    chosen = rule.get("latex") if latex else rule.get("name")
    if chosen:
        return str(chosen)
    arm = row.get("arm")
    return schema.ARM_NAMES.get(arm, str(arm) if arm else "?")


def series_label(row, fields=("arm",), paper: bool = True, latex: bool = False) -> str:
    """Legend label: the method name, plus whatever else `fields` asks for.

    `paper` is unused here; it stays for the signature figure.py calls with.
    """
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
