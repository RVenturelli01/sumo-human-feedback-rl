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
from rtplots import hparams as HP  # noqa: E402
from rtplots import formulas as FM  # noqa: E402
from rtplots import metrics as M  # noqa: E402
from rtplots import labels as L  # noqa: E402
from rtplots import rules, schema, selection, source, tikz  # noqa: E402
from rtplots.curves import aggregate as aggregate_curves  # noqa: E402
from rtplots.figure import (FigureSpec, Series, _apply_overrides,  # noqa: E402
                            _decollide, _merged_budget_dims, auto_hue,  # noqa: E402
                            merged_dims, split_panels)
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


def test_parse_group_riconosce_i_gruppi_della_campagna_finale():
    """thesis-final usa il prefisso th_: senza pattern, budget_level resta vuoto
    e quelle run sparirebbero dalle curve di budget senza alcun errore."""
    tag, level = source.parse_group("th_hybrid_soft_B1000")
    assert tag == "hybrid_soft" and level == 1000.0
    tag, level = source.parse_group("th_demo_only_B10")
    assert tag == "demo_only" and level == 10.0
    tag, level = source.parse_group("th_unw_bern_B100")
    assert tag == "unw_bern" and level == 100.0
    # niente budget in coda: non e' un gruppo di quella campagna
    assert source.parse_group("th_qualcosa") == (None, None)


def test_thesis_final_e_fra_i_progetti_indicizzati():
    from rtplots import paths
    assert "thesis-final" in paths.DEFAULT_PROJECTS


def test_parse_group_riconosce_anche_i_gruppi_gd():
    """thesis-grad-diagnostics scrive il livello come `_B<N>`, non `_<N>`."""
    tag, level = source.parse_group("gd_norm_on_p2_alpha_B100")
    assert tag == "norm_on_p2_alpha" and level == 100.0
    tag, level = source.parse_group("gd_baseline_norm_balance_no_norm_B1000")
    assert tag == "baseline_norm_balance_no_norm" and level == 1000.0


def test_fusion_solo_per_hybrid_e_norm_balance_quando_manca():
    """`gcl_fusion` non esiste nelle run precedenti agli schemi di fusione:
    li' il codice applicava il bilanciamento di norma, che e' il default."""
    assert source.derive_fusion({"gcl_fusion": "dual_adam_alpha"}, "hybrid") == "dual_adam_alpha"
    assert source.derive_fusion({}, "hybrid") == "norm_balance"
    # Su un braccio a sorgente singola non c'e' niente da fondere.
    assert source.derive_fusion({"gcl_fusion": "dual_adam_alpha"}, "demo") is None
    assert source.derive_fusion({}, "pref") is None


def test_fusion_distingue_bracci_che_arm_collasserebbe():
    """Baseline e prova 2 hanno lo stesso `arm`: senza `fusion` sarebbero
    indistinguibili nell'indice."""
    def cfg(fusion):
        return {"run": {"seed": 1},
                "algo": {"kwargs": {"total_queries": 100, "demo_weight": 1.0,
                                    "loss_type": "demo_2", "labels_type": "soft",
                                    "gcl_fusion": fusion}},
                "train": {"kwargs": {}}}
    base = source.row(fake_run(cfg("norm_balance"), group="gd_baseline_norm_balance_no_norm_B100"), "p")
    p2 = source.row(fake_run(cfg("dual_adam_alpha"), group="gd_p2_alpha_B100"), "p")
    assert base["arm"] == p2["arm"] == "hybrid_demo_2_soft"
    assert base["fusion"] != p2["fusion"]
    assert base["budget_level"] == p2["budget_level"] == 100.0


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


def test_una_metrica_che_nasce_a_meta_run_non_viene_estesa_all_indietro():
    """`np.interp` fuori dall'intervallo replica il primo valore.

    Le metriche `alpha/*` non esistono finche' i confronti non bastano a
    stimarne la dispersione: a B=10 comparivano dall'iterazione 44. Con la
    griglia che partiva da 0, le prime 44 iterazioni venivano riempite col
    primo valore vero, e il tratto piatto era indistinguibile da una misura.
    """
    df = make_index()
    run_ids = df[df.arm == "pref_soft"].run_id.tolist()
    curves = pd.DataFrame([
        {"run_id": rid, "step": float(s), "ret": 0.5 + 0.01 * (s - 44)}
        for rid in run_ids for s in range(44, 100)
    ])

    agg = aggregate_curves(curves, df, ["arm"], band="se")

    assert agg.step.min() == 44.0, "la curva non deve iniziare prima del primo dato"
    assert agg["mean"].iloc[0] == pytest.approx(0.5)


def test_i_seed_che_partono_in_ritardo_accorciano_da_sinistra_e_lo_segnalano():
    """Bordo sinistro come il destro: intersezione, non unione.

    Cosi' il numero di seed resta costante lungo la curva e la banda e'
    confrontabile punto per punto. Il taglio viene dichiarato, non subito.
    """
    df = make_index()
    run_ids = df[df.arm == "pref_soft"].run_id.tolist()
    inizi = dict(zip(run_ids, (44, 47, 50)))
    curves = pd.DataFrame([
        {"run_id": rid, "step": float(s), "ret": 1.0}
        for rid, primo in inizi.items() for s in range(primo, 100)
    ])

    agg = aggregate_curves(curves, df, ["arm"], band="se")

    assert agg.step.min() == 50.0            # l'ultimo che inizia
    assert (agg["n_seeds"] == 3).all()       # nessun punto con meno seed
    nota = agg.attrs["late_start"]
    assert len(nota) == 1 and nota[0]["start"] == 50.0 and nota[0]["earliest"] == 44.0


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


def _fusion_index() -> pd.DataFrame:
    """Stesso arm e stesso livello, due fusioni x due stati della normalizzazione.

    ``label_smoothing`` c'e' ma costante: la terza ablation esiste come colonna
    e deve restare muta finche' non varia davvero.
    """
    return pd.DataFrame([
        dict(run_id=f"r{i}", arm="hybrid_demo_2_soft", arm_family="hybrid",
             fusion=f, normalize_agent_reward=n, label_smoothing=0.0,
             budget_level=100.0, seed=1.0)
        for i, (f, n) in enumerate([("norm_balance", False), ("norm_balance", True),
                                    ("dual_adam_alpha", False), ("dual_adam_alpha", True)])
    ])


def _smoothing_index() -> pd.DataFrame:
    """Stesso arm, stessa fusione, stessa normalizzazione: cambia solo eps."""
    return pd.DataFrame([
        dict(run_id=f"s{i}", arm="hybrid_demo_2_bernoulli", arm_family="hybrid",
             fusion="norm_balance", normalize_agent_reward=False,
             label_smoothing=eps, budget_level=100.0, seed=1.0)
        for i, eps in enumerate([0.0, 0.1])
    ])


def test_compare_fusion_separa_gli_schemi_solo_se_attiva():
    df = _fusion_index()
    spento = FigureSpec(kind="budget")
    # Spento: il default delle curve di budget resta "un arm = una serie", e
    # entrambe le ablation vengono segnalate come mediate insieme.
    assert not spento.compare_fusion and not spento.compare_norm
    assert _merged_budget_dims(df, ["arm"], spento) == ["fusion", "normalize_agent_reward"]
    # Accese una per volta: resta segnalata solo l'altra.
    solo_fus = FigureSpec(kind="budget", compare_fusion=True)
    assert _merged_budget_dims(df, ["arm", "fusion"], solo_fus) == ["normalize_agent_reward"]
    solo_norm = FigureSpec(kind="budget", compare_norm=True)
    assert _merged_budget_dims(
        df, ["arm", "normalize_agent_reward"], solo_norm) == ["fusion"]
    # Entrambe: niente da segnalare.
    both = FigureSpec(kind="budget", compare_fusion=True, compare_norm=True)
    assert _merged_budget_dims(df, ["arm", "fusion", "normalize_agent_reward"], both) == []


def test_merges_fusion_tace_quando_non_ce_niente_da_mediare():
    df = _fusion_index()
    spec = FigureSpec(kind="budget")
    # Un solo valore presente: nessuna media silenziosa possibile.
    solo_una = df[(df.fusion == "norm_balance") & (~df.normalize_agent_reward)]
    assert _merged_budget_dims(solo_una, ["arm"], spec) == []
    # Su righe o colonne la dimensione separa gia' i pannelli.
    assert "fusion" not in _merged_budget_dims(
        df, ["arm"], FigureSpec(kind="budget", rows="fusion"))


def test_compare_smoothing_separa_gli_eps_solo_se_attiva():
    df = _smoothing_index()
    spento = FigureSpec(kind="budget")
    assert not spento.compare_smoothing
    # eps=0 e eps=0.1 nello stesso arm: senza la casella finiscono mediati.
    assert _merged_budget_dims(df, ["arm"], spento) == ["label_smoothing"]
    acceso = FigureSpec(kind="budget", compare_smoothing=True)
    assert _merged_budget_dims(df, ["arm", "label_smoothing"], acceso) == []
    # Costante: niente da segnalare, anche a casella spenta.
    assert "label_smoothing" not in _merged_budget_dims(_fusion_index(), ["arm"], spento)


def test_senza_smoothing_prende_il_tratto_continuo():
    """Convenzione condivisa con la normalizzazione: l'ablation e' la tratteggiata."""
    df = _smoothing_index()
    styles = {"a": {"color": "C0", "style": "solid"}, "b": {"color": "C0", "style": "solid"}}
    matches = {"a": {"label_smoothing": 0.0}, "b": {"label_smoothing": 0.1}}
    # L'ordine arriva gia' crescente da _sort_ascending: eps=0 e' il primo.
    _decollide(styles, ["a", "b"], matches)
    assert styles["a"]["style"] == "solid"
    assert styles["b"]["style"] != "solid"
    assert set(df.label_smoothing) == {0.0, 0.1}


def test_i_flag_sopravvivono_al_salvataggio():
    """Sono parte dello spec, quindi --runs-file rifa' la stessa figura."""
    spec = FigureSpec(kind="budget", compare_fusion=True, compare_norm=True,
                      compare_smoothing=True)
    back = FigureSpec.from_dict(spec.to_dict())
    assert back.compare_fusion is True and back.compare_norm is True
    assert back.compare_smoothing is True


def test_serie_dello_stesso_braccio_non_restano_indistinguibili():
    """Le regole di style.toml colorano per `arm`: due configurazioni dello
    stesso braccio (normalizzazione on/off, schemi di fusione, ...) uscirebbero
    identiche. Il colore resta quello della regola, a separarle e' il tratto."""
    from rtplots.figure import _decollide
    styles = {
        "baseline, no-norm": {"color": "#e34948", "style": "solid"},
        "baseline, norm": {"color": "#e34948", "style": "solid"},
    }
    # In legenda i booleani vanno con "si" per primo: l'ordine da solo darebbe
    # il tratto continuo alla configurazione normalizzata.
    order = ["baseline, norm", "baseline, no-norm"]
    matches = {"baseline, norm": {"normalize_agent_reward": True},
               "baseline, no-norm": {"normalize_agent_reward": False}}
    _decollide(styles, order, matches)
    assert styles["baseline, no-norm"]["color"] == styles["baseline, norm"]["color"]
    # La condizione disattivata resta continua, l'ablation e' la tratteggiata,
    # sempre nello stesso verso in ogni figura.
    assert styles["baseline, no-norm"]["style"] == "solid"
    assert styles["baseline, norm"]["style"] == "dashed"


def test_decollide_non_tocca_le_serie_gia_distinguibili():
    """soft e bernoulli condividono il colore ma hanno gia' tratti diversi."""
    from rtplots.figure import _decollide
    styles = {
        "hybrid (soft)": {"color": "#e34948", "style": "solid"},
        "hybrid (bernoulli)": {"color": "#e34948", "style": "dashed"},
        "demo_2": {"color": "#008300", "style": "solid"},
    }
    before = {k: v["style"] for k, v in styles.items()}
    _decollide(styles, list(styles))
    assert {k: v["style"] for k, v in styles.items()} == before


# --- formule ----------------------------------------------------------------

def test_ogni_metrica_dei_gradienti_ha_la_sua_formula():
    """Il pannello delle definizioni non deve restare vuoto proprio sulle
    metriche per cui e' stato aggiunto."""
    documented = [k for group, items in M.METRIC_GROUPS
                  if "Gradienti" in group or "Normalizzazione" in group
                  or "Stima di α" in group
                  for k, *_ in items]
    assert documented, "i tre gruppi di metriche esistono ancora"
    assert [k for k in documented if k not in FM.METRIC_FORMULAS] == []


def test_le_metriche_di_alpha_sono_nella_tendina_delle_curve():
    """Il gruppo deve arrivare davvero alla tendina COSA PLOTTARE.

    Passa da ``ui_groups("curve")``: se una chiave fosse registrata come
    ``summary`` sparirebbe dal pannello della curva di apprendimento senza che
    nulla si lamenti.
    """
    gruppi = {g["group"]: [o["key"] for o in g["options"]]
              for g in M.ui_groups("curve")}
    alpha = next((v for k, v in gruppi.items() if "Stima di α" in k), None)
    assert alpha is not None, "il gruppo α manca dalla tendina delle curve"

    attese = {f"alpha/{k}_{c}"
              for k in ("V", "S", "cv2", "gradmean_norm_sq", "n", "batch")
              for c in ("pref", "demo")}
    assert attese <= set(alpha)
    assert "reward/hybrid_alpha" in alpha

    # Asse x giusto: sono loggate dall'algoritmo, non dall'agente SAC.
    for key in attese:
        assert M.metric_info(key)["step_key"] == M.ITER_STEP


def test_ogni_fusione_implementata_ha_la_sua_equazione():
    from rtplots.schema import FUSION_NAMES
    storiche = {"dual_adam_reliability", "demo_anchor_inv_var"}
    attuali = set(FUSION_NAMES) - storiche
    assert attuali <= set(FM.FUSION_FORMULAS)


def test_blocks_include_le_fusioni_in_selezione():
    blocks = FM.blocks("reward/hybrid_alpha", ["dual_adam_alpha", "norm_balance"])
    assert [b["title"] for b in blocks][0] == "Peso delle dimostrazioni"
    fusion_block = blocks[1]["lines"]
    assert any("prova 2" in line for line in fusion_block)
    assert any("norm_balance" in line for line in fusion_block)
    # Una fusione sconosciuta non deve far comparire un blocco vuoto.
    assert len(FM.blocks("sweep/mean_fast_return", ["mai_vista"])) == 0


def test_salvataggi_diversi_non_si_sovrascrivono(tmp_path, monkeypatch):
    """Due nomi che collassano sullo stesso slug devono restare due file."""
    # In produzione selection.json sta in CACHE_DIR e le selezioni in
    # CACHE_DIR/selections: cartelle diverse, altrimenti listing() lo conterebbe.
    store = tmp_path / "selections"; store.mkdir()
    monkeypatch.setattr(selection, "SELECTIONS_DIR", store)
    monkeypatch.setattr(selection, "SELECTION_JSON", tmp_path / "selection.json")
    base = selection.slugify("selezione 03/08 19:13")
    for name in ("selezione 03/08 19:13", "selezione 03/08 19:13 (bis)"):
        slug = selection.free_slug(selection.slugify(name), name)
        selection.write({"version": 1, "name": name, "slug": slug, "saved_at": name,
                         "n_runs": 1, "run_ids": [], "spec": {}})
    assert len(selection.listing()) == 2
    # Risalvare con lo STESSO nome aggiorna, non duplica.
    slug = selection.free_slug(base, "selezione 03/08 19:13")
    assert slug == base
    selection.write({"version": 1, "name": "selezione 03/08 19:13", "slug": slug,
                     "saved_at": "dopo", "n_runs": 9, "run_ids": [], "spec": {}})
    assert len(selection.listing()) == 2
    assert [i["n_runs"] for i in selection.listing() if i["slug"] == base] == [9]


def test_budget_level_e_una_dimensione_di_griglia():
    """Una riga per B: budget_level e' gia' la B unificata (10 preferenze + 10
    traiettorie per l'ibrido, solo una delle due per i bracci a sorgente
    singola), quindi deve essere selezionabile come riga/colonna."""
    assert "budget_level" in schema.GRID_FIELDS
    assert schema.panel_title("budget_level", 100.0) == "B = 100"
    # Non in sidebar: il filtro si fa con query_budget/demo_budget.
    assert "budget_level" not in schema.UI_DIMENSIONS


def test_formule_stesso_contenuto_in_svg_e_raster():
    """L'immagine esportata e il pannello a schermo vengono dalla stessa
    sorgente: se divergessero, il report mostrerebbe formule diverse."""
    blocks = FM.blocks("reward/hybrid_alpha", ["dual_adam_alpha"])
    svg = FM.render_svg(blocks)
    png = FM.render_png(blocks, dpi=100)
    assert svg.startswith("<svg")
    assert png[:4] == b"\x89PNG"


def test_curve_segnala_quando_una_run_corta_accorcia_la_serie():
    """La griglia comune si ferma alla run piu' corta del gruppo: prima lo
    faceva in silenzio e la curva sembrava interrotta senza motivo."""
    curves = pd.concat([
        pd.DataFrame({"run_id": "a", "step": [0.0, 1e6, 2e6], "ret": [0.0, 5.0, 10.0]}),
        pd.DataFrame({"run_id": "b", "step": [0.0, 1e6, 2e6], "ret": [0.0, 5.0, 10.0]}),
        pd.DataFrame({"run_id": "c", "step": [0.0, 5e5], "ret": [0.0, 3.0]}),
    ], ignore_index=True)
    meta = pd.DataFrame({"run_id": ["a", "b", "c"], "arm": ["demo_2"] * 3})
    agg = aggregate_curves(curves, meta, ["arm"])
    trunc = agg.attrs["truncated"]
    assert len(trunc) == 1
    assert trunc[0]["run_id"] == "c"
    assert trunc[0]["end"] == 5e5 and trunc[0]["longest"] == 2e6
    # Con run di lunghezza simile non si segnala niente.
    same = curves[curves.run_id != "c"]
    assert aggregate_curves(same, meta[meta.run_id != "c"], ["arm"]).attrs["truncated"] == []


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
    # In produzione selection.json sta in CACHE_DIR e le selezioni in
    # CACHE_DIR/selections: cartelle diverse, altrimenti listing() lo conterebbe.
    store = tmp_path / "selections"; store.mkdir()
    monkeypatch.setattr(selection, "SELECTIONS_DIR", store)
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


# --- iperparametri di una riga di copertura ---------------------------------

def fake_config(seed: int, **over) -> dict:
    cfg = {
        "run": {"seed": seed, "n_expert_trajectories": 500, "output_dir": "outputs/x",
                "name": f"run-{seed}", "group": "g"},
        "algo": {"kwargs": {"lr_rew": 0.0011542956981980379, "l2_rew": 1.1265e-06,
                            "gradient_steps_rew": 139, "batch_size_expert": 64,
                            "batch_size_model": 64, "batch_size_pref": 256,
                            "reward_model_kwargs": {"n_ensembles": 1, "net_arch": [64, 64]}}},
        "wandb": {"project": "p", "tags": ["a", "b"]},
    }
    cfg["algo"]["kwargs"].update(over)
    return cfg


def loader_from(configs):
    """Loader iniettabile con la firma di `hparams.load_config`."""
    def load(run_id, project, state):
        if run_id not in configs:
            raise KeyError(run_id)
        return configs[run_id]
    return load


def test_appiattimento_tiene_le_liste_intere():
    flat = HP.flatten(fake_config(1))
    assert flat["algo.kwargs.lr_rew"] == 0.0011542956981980379
    assert flat["algo.kwargs.reward_model_kwargs.net_arch"] == [64, 64]
    assert "algo.kwargs.reward_model_kwargs.net_arch.0" not in flat


def test_solo_il_seed_distingue_le_run_di_un_gruppo_sano():
    per_run = {f"r{s}": HP.run_hparams(fake_config(s)) for s in (1, 2, 3)}
    common, differing = HP.split_common(per_run)
    assert list(differing) == ["run.seed"]
    assert common["algo.kwargs.gradient_steps_rew"] == 139


def test_una_chiave_presente_in_una_sola_run_conta_come_differenza():
    per_run = {"r1": HP.run_hparams(fake_config(1)),
               "r2": HP.run_hparams(fake_config(1, label_smoothing=0.1))}
    common, differing = HP.split_common(per_run)
    assert "algo.kwargs.label_smoothing" in differing
    assert "algo.kwargs.label_smoothing" not in common


def test_il_float_non_perde_cifre():
    # lr_rew deve restare confrontabile con l'export di Optuna
    assert "0.0011542956981980379" in HP.to_yaml([(None, "a", {"lr": 0.0011542956981980379})])


def test_yaml_scrive_booleani_e_liste_in_sintassi_yaml():
    text = HP.to_yaml([(None, "a", {"b": True, "c": None, "d": [64, 64], "e": "x y"})])
    assert "b: true" in text and "c: null" in text
    assert "d: [64, 64]" in text and 'e: "x y"' in text


def test_yaml_del_gruppo_separa_comune_e_differenze():
    configs = {f"r{s}": fake_config(s) for s in (1, 2, 3)}
    text = HP.group_yaml([{"run_id": f"r{s}", "project": "p", "state": "finished",
                           "name": f"n{s}"} for s in (1, 2, 3)],
                         cells=["hybrid_demo_2_soft"], columns=["Algoritmo"],
                         loader=loader_from(configs))
    assert "gruppo:" in text and "comune:" in text and "differenze:" in text
    assert "Algoritmo: hybrid_demo_2_soft" in text
    assert "algo.kwargs.batch_size_expert: 64" in text
    assert "run.seed [r1]: 1" in text
    # le chiavi che dicono solo dove la run e' finita non sono iperparametri
    assert "run.output_dir" not in text and "wandb.project" not in text


def test_una_run_illeggibile_non_blocca_le_altre():
    text = HP.group_yaml(
        [{"run_id": "r1", "project": "p", "state": "finished"},
         {"run_id": "assente", "project": "p", "state": "finished"}],
        loader=loader_from({"r1": fake_config(1)}))
    assert "errori:" in text and "assente" in text
    assert "algo.kwargs.gradient_steps_rew: 139" in text


def test_handler_hparams_rifiuta_run_fuori_dall_indice():
    df = make_index()
    assert "error" in api.hparams(df, {"run_ids": []})
    res = api.hparams(df, {"run_ids": ["non-esiste"]})
    assert "non-esiste" in res["error"]


def test_handler_hparams_si_ferma_prima_di_troppe_richieste():
    df = make_index()
    res = api.hparams(df, {"run_ids": list(df.run_id) * 2})
    assert "massimo" in res["error"]


def test_handler_hparams_usa_le_run_della_riga(monkeypatch):
    df = make_index()
    ids = ["hybrid_demo_2_soft-1", "hybrid_demo_2_soft-2", "hybrid_demo_2_soft-3"]
    configs = {rid: fake_config(i + 1) for i, rid in enumerate(ids)}
    monkeypatch.setattr(HP, "load_config", loader_from(configs))
    res = api.hparams(df, {"run_ids": ids, "cells": ["soft"], "columns": ["Etichette"]})
    assert res["n_runs"] == 3
    assert res["filename"] == "hparams_budget_hybrid_demo_2_soft_1000.yaml"
    assert "Etichette: soft" in res["yaml"]
    assert "algo.kwargs.lr_rew: 0.0011542956981980379" in res["yaml"]


def test_ogni_riga_di_copertura_porta_le_sue_run():
    # il bottone della pagina manda esattamente questi id all'handler
    df = make_index()
    rows = api.query(df, {"dims": {}})["coverage"]["rows"]
    assert rows and all(r["run_ids"] for r in rows)
    assert all(set(r["run_ids"]) <= set(df.run_id) for r in rows)


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
