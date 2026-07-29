"""Test del toolkit dei grafici (niente rete, niente W&B).

    .venv/bin/python -m pytest plots/tests -q

Coprono le parti che servono a tutte le figure: derivazione dell'arm dalla
config Hydra (il punto piu' delicato: e' quello che risponde a "che algoritmo
sta girando questa run", vedi `rtplots/source.py`), formattazione dello schema,
scelta automatica delle serie (incluso il caso speciale delle curve di
budget, dove il budget non deve mai diventare una dimensione di colore),
aggregazione (curve nel tempo e curve di budget), regole di style.toml,
handler del selettore ed export LaTeX.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rtplots import budget as B  # noqa: E402
from rtplots import labels as L  # noqa: E402
from rtplots import rules, schema, selection, source, tikz  # noqa: E402
from rtplots.curves import aggregate as aggregate_curves  # noqa: E402
from rtplots.figure import (FigureSpec, Series, _apply_overrides,  # noqa: E402
                            auto_hue, merged_dims, split_panels)
from rtplots.webui import api  # noqa: E402


# --- indice finto -----------------------------------------------------------

def make_index() -> pd.DataFrame:
    rows = []
    specs = [
        ("pref_soft", "pref", None, "soft", 10000, None),
        ("pref_bernoulli", "pref", None, "bernoulli", 100000, None),
        ("demo_1", "demo", "demo_1", None, None, 500),
        ("demo_2", "demo", "demo_2", None, None, 500),
        ("hybrid_demo_2_soft", "hybrid", "demo_2", "soft", 1000, 500),
        ("hybrid_demo_2_bernoulli", "hybrid", "demo_2", "bernoulli", 1000, 500),
    ]
    for arm, family, demo_loss, pref_labels, qbud, dbud in specs:
        for seed in (1, 2, 3):
            rows.append(dict(
                run_id=f"{arm}-{seed}", name=f"{arm}_{seed}", group=f"budget_{arm}_{qbud or dbud}",
                state="finished", project="tuning-thesis-budget-curves-completion",
                arm=arm, arm_family=family, demo_loss=demo_loss, pref_labels=pref_labels,
                demo_mode="gcl", query_budget=qbud, demo_budget=dbud,
                budget_level=float(qbud or dbud), normalize_agent_reward=True,
                initial_queries=(qbud or 0) * 0.1 if qbud else None,
                initial_queries_frac=0.1 if qbud else None, demo_weight=0.0 if family == "pref" else 1.0,
                query_schedule="constant", fragmenter_type="active", pref_temperature=20.0,
                reward_net_arch="[64, 64]", demo_subsample_seed=None, total_timesteps=2e6,
                seed=seed,
            ))
    return pd.DataFrame(rows)


def make_curves(index: pd.DataFrame) -> pd.DataFrame:
    """Una curva sintetica (2 punti) per ogni run dell'indice finto."""
    frames = []
    for run_id in index.run_id:
        frames.append(pd.DataFrame({"run_id": run_id, "step": [0.0, 1e6], "ret": [0.0, 10.0]}))
    return pd.concat(frames, ignore_index=True)


# --- schema -------------------------------------------------------------------

def test_html_value_non_confonde_uno_con_vero():
    # in Python 1.0 == True: budget_level=1 non deve diventare "si'"
    assert schema.html_value("budget_level", 1.0) == "1"
    assert schema.html_value("normalize_agent_reward", True) == "sì"
    # demo_budget mancante e' "dataset intero" (nessun sottocampionamento), non
    # un valore ignoto: un campo generico invece mostra il trattino
    assert schema.html_value("demo_budget", float("nan")) == "dataset intero"
    assert schema.html_value("pref_temperature", float("nan")) == "—"
    assert schema.html_value("arm", "pref_soft") == "pref_soft"


def test_panel_title_e_legenda_dallo_stesso_campo():
    assert schema.panel_title("query_budget", 5000.0) == "query = 5000"
    assert schema.legend_bit("query_budget", 5000.0) == "5000 query"
    assert schema.legend_bit("demo_budget", float("nan")) is None
    assert schema.legend_bit("mai_visto", 3) is None


def test_arm_e_lunica_dimensione_di_identita_dellalgoritmo():
    # "arm" da solo distingue gia' le 8 combinazioni (4 bracci base + 4 di
    # hybrid): demo_loss/pref_labels/arm_family sono ridondanti con lui e non
    # sono piu' dimensioni di serie ne' di UI/griglia, altrimenti finiscono per
    # separare le curve una seconda volta sulla stessa informazione.
    assert "arm" in schema.SERIES_FIELDS
    assert "demo_loss" not in schema.SERIES_FIELDS
    assert "pref_labels" not in schema.SERIES_FIELDS
    assert "arm_family" not in schema.SERIES_FIELDS
    assert "seed" not in schema.SERIES_FIELDS   # il seed si aggrega, non separa


# --- derivazione dell'arm dalla config Hydra ---------------------------------

def test_derive_arm_pref_only():
    algo = {"total_queries": 10000, "demo_weight": 0.0, "labels_type": "soft"}
    bits = source.derive_arm(algo, {})
    assert bits == {"arm": "pref_soft", "arm_family": "pref", "demo_loss": None,
                    "pref_labels": "soft"}


def test_derive_arm_pref_bernoulli():
    algo = {"total_queries": 100000, "demo_weight": 0.0, "labels_type": "binary_bernoulli"}
    assert source.derive_arm(algo, {})["arm"] == "pref_bernoulli"


def test_derive_arm_demo_only():
    algo = {"total_queries": 0, "demo_weight": 1.0, "loss_type": "demo_2"}
    bits = source.derive_arm(algo, {})
    assert bits == {"arm": "demo_2", "arm_family": "demo", "demo_loss": "demo_2",
                    "pref_labels": None}


def test_derive_arm_hybrid_tutte_le_combinazioni():
    for loss in ("demo_1", "demo_2"):
        for labels, expected in (("soft", "soft"), ("binary_bernoulli", "bernoulli")):
            algo = {"total_queries": 5000, "demo_weight": 0.6, "loss_type": loss,
                    "labels_type": labels}
            bits = source.derive_arm(algo, {})
            assert bits["arm"] == f"hybrid_{loss}_{expected}"
            assert bits["arm_family"] == "hybrid"


def test_derive_arm_senza_preferenze_ne_demo_e_sconosciuto():
    bits = source.derive_arm({"total_queries": 0, "demo_weight": 0.0}, {})
    assert bits["arm"] is None and bits["arm_family"] is None


def test_parse_group_livello_e_tag():
    tag, level = source.parse_group("budget_hybrid_demo_2_bern_hom_5446")
    assert tag == "hybrid_demo_2_bern_hom" and level == 5446.0
    tag, level = source.parse_group("budget_pref_soft_10000")
    assert tag == "pref_soft" and level == 10000.0
    assert source.parse_group("tune_pref_soft") == (None, None)
    assert source.parse_group(None) == (None, None)


def fake_run(config, group=None, tags=(), rid="abc", state="finished"):
    return SimpleNamespace(id=rid, name="run", group=group, state=state,
                           tags=list(tags), created_at="2026-01-01", config=config)


def test_row_legge_una_run_hybrid_completa():
    cfg = {
        "run": {"seed": 2, "n_expert_trajectories": 500, "demo_subsample_seed": None},
        "algo": {"kwargs": {
            "total_queries": 1000, "demo_weight": 0.6, "loss_type": "demo_2",
            "labels_type": "soft", "initial_queries": 100, "query_schedule": "constant",
            "fragmenter_type": "active", "normalize_agent_reward": True,
            "relabel_rewards": True, "pref_temperature": 20.0,
            "reward_model_kwargs": {"net_arch": [64, 64]},
        }},
        "train": {"kwargs": {"total_timesteps": 2000000}},
    }
    row = source.row(fake_run(cfg, group="budget_hybrid_demo_2_soft_hom_1000"), "tuning-thesis-budget-curves-completion")
    assert row["arm"] == "hybrid_demo_2_soft"
    assert row["query_budget"] == 1000.0 and row["demo_budget"] == 500.0
    assert row["budget_level"] == 1000.0 and row["group_tag"] == "hybrid_demo_2_soft_hom"
    assert row["initial_queries_frac"] == 0.1
    assert row["normalize_agent_reward"] is True
    assert row["reward_net_arch"] == "[64, 64]"
    assert row["seed"] == 2.0


def test_row_demo_budget_none_e_dataset_intero_non_un_valore_mancante():
    cfg = {"run": {"seed": 1, "n_expert_trajectories": None},
           "algo": {"kwargs": {"total_queries": 0, "demo_weight": 1.0, "loss_type": "demo_1"}},
           "train": {"kwargs": {}}}
    row = source.row(fake_run(cfg), "p")
    assert row["demo_budget"] is None
    assert schema.html_value("demo_budget", row["demo_budget"]) == "dataset intero"


# --- scelta delle serie -----------------------------------------------------

def test_auto_hue_separa_gli_arm():
    df = make_index()
    assert auto_hue(df) == ["arm"]


def test_merged_dims_segnala_solo_cio_che_non_e_su_hue_righe_colonne():
    # merged_dims non indovina le ridondanze (quello lo fa auto_hue): segnala
    # ogni colonna di serie che varia e non e' ne' su hue ne' su righe/colonne,
    # anche se e' perfettamente correlata con cio' che gia' c'e' su hue.
    df = pd.DataFrame({
        "run_id": ["a", "b"], "arm": ["pref_soft", "pref_bernoulli"],
        "query_budget": [10000.0, 100000.0],
    })
    assert merged_dims(df, ["arm"]) == ["query_budget"]
    assert merged_dims(df, ["arm", "query_budget"]) == []
    assert merged_dims(df, [], panels=("arm", "query_budget")) == []


def test_auto_hue_ignora_le_dimensioni_su_righe_e_colonne():
    df = make_index()
    assert "arm" not in auto_hue(df, exclude=("arm", None))


# --- aggregazione: curve di apprendimento -----------------------------------

def test_aggregate_curve_media_sui_seed():
    df = make_index()
    curves = make_curves(df)
    agg = aggregate_curves(curves, df, ["arm"], band="se")
    one = agg[agg.arm == "pref_soft"]
    assert set(one.step) == {0.0, 1e6}
    assert one[one.step == 1e6]["mean"].iloc[0] == 10.0
    assert (one["n_seeds"] == 3).all()


# --- aggregazione: curve di budget ------------------------------------------

def test_budget_aggregate_un_punto_per_livello(monkeypatch, tmp_path):
    df = make_index()
    fake_summaries = pd.DataFrame({
        "run_id": df.run_id,
        "sweep/mean_fast_return": list(range(len(df))),
    })
    monkeypatch.setattr(B, "load_summaries", lambda index, **kw: fake_summaries)
    agg = B.aggregate(df, ["arm"], "budget_level", "sweep/mean_fast_return")
    # un solo livello di budget per arm in questo indice finto: una riga a serie
    assert set(agg.arm) == set(df.arm.unique())
    assert (agg.n_seeds == 3).all()


def test_minimum_budget_regola_del_90_percento():
    levels = pd.Series([100, 500, 1000, 5000])
    # relativo al peggiore (100->10.0): 500 e' all'87.5%, sotto soglia;
    # 1000 e' al 95% e anche il successivo (5000, 100%) passa -> minimo 1000
    metric = pd.Series([10.0, 45.0, 48.0, 50.0])
    assert B.minimum_budget(levels, {"m": metric}) == 1000
    # se anche 500 passa la soglia, e il livello dopo (1000) passa a sua volta,
    # il minimo scende a 500
    metric2 = pd.Series([10.0, 47.0, 48.0, 50.0])
    assert B.minimum_budget(levels, {"m": metric2}) == 500
    assert B.minimum_budget(pd.Series([100]), {"m": pd.Series([1.0])}) is None


# --- spec e selezioni --------------------------------------------------------

def test_spec_round_trip():
    spec = FigureSpec(kind="budget", rows="pref_labels", cols=None, hue=["arm"],
                      budget_x="query_budget")
    back = FigureSpec.from_dict(spec.to_dict())
    assert back.kind == "budget" and back.rows == "pref_labels"
    assert back.hue == ["arm"] and back.budget_x == "query_budget"


def test_spec_default_metric_dipende_dal_kind():
    assert FigureSpec(kind="curve").metric.startswith("agent/")
    assert FigureSpec(kind="budget").metric.startswith("sweep/")


def test_selezione_lettura_scrittura(tmp_path, monkeypatch):
    monkeypatch.setattr(selection, "SELECTIONS_DIR", tmp_path)
    monkeypatch.setattr(selection, "SELECTION_JSON", tmp_path / "selection.json")
    spec = FigureSpec(kind="budget", filters=["arm_family=pref"])
    payload = {"name": "prova", "slug": "prova", "saved_at": "2026-07-29T10:00:00",
              "n_runs": 6, "filter_args": ["arm_family=pref"], "dims": {}, "seeds": {},
              "excluded": [], "run_ids": ["a", "b"], "spec": spec.to_dict(),
              "version": 1}
    stored = selection.write(payload)
    assert stored.exists()
    loaded_spec, data = selection.spec_from(stored)
    assert loaded_spec.kind == "budget" and loaded_spec.run_ids == ["a", "b"]
    assert data["name"] == "prova"


# --- handler del selettore ---------------------------------------------------

def test_query_conta_run_configurazioni_e_copertura():
    df = make_index()
    res = api.query(df, {"dims": {"arm_family": {"op": "in", "values": ["hybrid"]}}})
    assert res["n_runs"] == 6
    assert res["n_configs"] == 2                     # soft e bernoulli
    assert res["filter_args"] == ["arm_family=hybrid"]
    assert "Algoritmo" in res["coverage"]["columns"]


def test_operatori_dei_filtri():
    df = make_index()
    q = lambda dims: api.query(df, {"dims": dims})
    assert q({"arm_family": {"op": "is", "values": ["pref"]}})["n_runs"] == 6
    assert q({"arm_family": {"op": "is_not", "values": ["pref"]}})["n_runs"] == 12
    assert q({"arm_family": ["pref"]})["n_runs"] == 6   # lista nuda = "fra"


def test_filtri_negativi_tradotti_in_sintassi_cli():
    df = make_index()
    args = api.query(df, {"dims": {"arm_family": {"op": "not_in", "values": ["pref"]}}})["filter_args"]
    assert args == ["arm_family!=pref"]


def test_spec_dalla_pagina():
    df = make_index()
    payload = {"dims": {}, "grid": {"kind": "budget", "rows": "pref_labels", "cols": "",
                                    "hue": ["arm"], "band": "iqr",
                                    "metric": "sweep/mean_fast_return", "budget_x": "demo_budget"}}
    spec = api.spec_from_payload(payload, df)
    assert spec.kind == "budget" and spec.rows == "pref_labels" and spec.cols is None
    assert spec.band == "iqr" and spec.budget_x == "demo_budget"
    assert spec.state == "any" and len(spec.run_ids) == len(df)
    assert spec.row_captions == "auto"


def test_esclusioni_tolgono_run_ma_non_righe_di_copertura():
    df = make_index()
    victim = df.run_id.iloc[0]
    res = api.query(df, {"dims": {}, "excluded": [victim]})
    assert res["n_excluded"] == 1
    rows = res["coverage"]["rows"]
    assert any(r["n_kept"] < r["n_runs"] for r in rows)


# --- regole scritte a mano (style.toml) -------------------------------------

@pytest.fixture
def regole(tmp_path, monkeypatch):
    def write(text: str):
        path = tmp_path / "style.toml"
        path.write_text(text)
        monkeypatch.setattr(rules, "RULES_FILE", path)
        rules.load(force=True)
        return path
    yield write
    rules.load(force=True)


def test_vince_la_prima_regola_che_combacia(regole):
    regole("""
[[series]]
match = { arm = "hybrid_demo_2_soft" }
color = "#111111"
[[series]]
match = { arm_family = "hybrid" }
color = "#222222"
""")
    assert rules.rule_for({"arm": "hybrid_demo_2_soft", "arm_family": "hybrid"})["color"] == "#111111"
    assert rules.rule_for({"arm": "hybrid_demo_2_bernoulli", "arm_family": "hybrid"})["color"] == "#222222"
    assert rules.rule_for({"arm": "pref_soft"}) == {}
    assert rules.rule_for({"arm_family": "hybrid"})["color"] == "#222222"


def test_nomi_delle_serie_presi_dal_file(regole):
    regole("""
[[series]]
match = { arm = "pref_soft" }
name = 'pref_soft'
latex = '\\prefsoft'
""")
    row = {"arm": "pref_soft"}
    assert L.series_label(row, fields=("arm",)) == "pref_soft"
    assert L.series_label(row, fields=("arm",), latex=True) == r"\prefsoft"


def test_senza_regola_il_nome_resta_quello_del_codice(regole):
    regole("")
    row = {"arm": "hybrid_demo_1_soft"}
    assert L.series_label(row, fields=("arm",)) == r"$\mathtt{hybrid\_demo\_1\ (soft)}$"


def test_file_rotto_non_rompe_i_grafici(regole, capsys):
    regole("questo non e' TOML [[[")
    assert rules.series_rules() == []
    assert rules.get("lines", "width") == 1.4
    assert "non e' TOML valido" in capsys.readouterr().out


def test_opzioni_pgfplots_in_coda_alle_altre():
    code = ("\\begin{axis}[\ntick pos=left,\nxmin=0, xmax=1\n]\n"
            "\\addplot table {};\n\\end{axis}\n")
    out = tikz._add_axis_options(code, ["width=\\figurewidth", " ", "ymin=-11"])
    axis = out.split("]\n")[0]
    assert axis.endswith("xmin=0, xmax=1,\nwidth=\\figurewidth,\nymin=-11\n")
    assert tikz._add_axis_options(code, []) == code


# --- ritocchi fatti a mano nell'anteprima -----------------------------------

def test_ritocco_rinomina_e_ricolora_in_tutti_i_pannelli():
    series = make_series()
    styles = dict(series.styles)
    styles["A"] = {**styles["A"], "latex": r"\prefsoft"}
    agg, order, styles, matches = _apply_overrides(
        series.agg, series.order, styles, {"A": {"arm": "pref_soft"}, "B": {}},
        {"A": {"name": "pref_soft: stale", "color": "#785EF0"}})
    assert order == ["pref_soft: stale", "B"]
    assert "A" not in set(agg.label)
    assert styles["pref_soft: stale"]["color"] == "#785EF0"
    assert styles["pref_soft: stale"]["latex"] is None
    assert matches["pref_soft: stale"] == {"arm": "pref_soft"}


def test_ritocco_di_una_serie_sparita_viene_ignorato():
    series = make_series()
    agg, order, styles, _ = _apply_overrides(
        series.agg, series.order, dict(series.styles), {"A": {}, "B": {}},
        {"Z": {"name": "mai vista"}})
    assert order == ["A", "B"] and "mai vista" not in styles


def test_regola_da_incollare_in_style_toml():
    snippet = api.rule_snippet(
        "pref_soft: stale", {"arm": "pref_soft", "budget_level": 1000.0},
        {"color": "#2a78d6", "latex": "\\prefsoft"})
    assert snippet.splitlines()[0] == "[[series]]"
    assert 'match = { arm = "pref_soft", budget_level = 1000 }' in snippet
    assert "latex = '\\prefsoft'" in snippet
    assert "name  = 'pref_soft: stale'" in snippet


# --- export LaTeX -----------------------------------------------------------

def make_series() -> Series:
    """Griglia 2x2 (arm_family x pref_labels) gia' aggregata, due serie."""
    rows = []
    for fam in ("pref", "hybrid"):
        for labels in ("soft", "bernoulli"):
            for label in ("A", "B"):
                for step in (0.0, 1.0):
                    rows.append(dict(step=step, mean=1.0, lo=0.0, hi=2.0, n_seeds=3,
                                     label=label, arm_family=fam, pref_labels=labels))
    styles = {lab: {"color": color, "width": 1.4, "style": "solid",
                    "band_alpha": 0.18, "latex": None}
              for lab, color in (("A", "#2a78d6"), ("B", "#e34948"))}
    return Series(sel=pd.DataFrame(), agg=pd.DataFrame(rows), order=["A", "B"],
                  styles=styles, hue=["arm"], matches={"A": {}, "B": {}},
                  ylabel="y", metric_label="Mean return", merged=[])


def test_split_panels_da_un_pannello_per_riquadro():
    spec = FigureSpec(rows="arm_family", cols="pref_labels")
    panels = split_panels(make_series(), spec)
    assert len(panels) == 4
    assert all(p.spec.rows is None and p.spec.cols is None for p in panels)
    assert set(panels[0].series.agg.arm_family) == {"hybrid"} or \
        set(panels[0].series.agg.arm_family) == {"pref"}


def test_split_panels_senza_griglia_da_un_pannello_solo():
    panels = split_panels(make_series(), FigureSpec())
    assert len(panels) == 1 and panels[0].slug == "" and panels[0].caption == ""


def test_snippet_latex_monta_una_subfigure_per_pannello():
    panels = split_panels(make_series(), FigureSpec(rows="arm_family", cols="pref_labels"))
    names = [f"fig_{p.slug}.tex" for p in panels]
    snippet = api.tex_snippet("fig", panels, names, ncol=2, caption="Didascalia.")
    assert snippet.count("\\begin{subfigure}") == 4
    assert "subcaption" in snippet
    assert "\\label{fig:fig}" in snippet


def test_tikzplotlib_importabile_con_gli_alias():
    # tikzplotlib 0.10.1 non regge matplotlib 3.6+/numpy 2/webcolors 24+ senza
    # gli alias di rtplots.tikz: se questo test fallisce, l'export LaTeX e' rotto.
    if importlib.util.find_spec("tikzplotlib") is None:
        pytest.skip("tikzplotlib non installato")
    assert tikz.unavailable_reason() is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
