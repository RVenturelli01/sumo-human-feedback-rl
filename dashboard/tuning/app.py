"""Interactive dashboard for the thesis tuning studies."""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any

import numpy as np
import plotly.graph_objects as go
from dash import (
    Dash,
    Input,
    Output,
    callback,
    dash_table,
    dcc,
    html,
    no_update,
)

from analytics import (
    all_params,
    candidate_table,
    distance_matrix,
    parameter_association,
    scored,
    summary,
)
from data import ALGORITHMS, load_cache, sync_data


GRAPH_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "tuning-thesis",
        "width": 1800,
        "height": 1000,
        "scale": 2,
    },
}
PLOT_LAYOUT = {
    "paper_bgcolor": "#ffffff",
    "plot_bgcolor": "#ffffff",
    "font": {
        "family": "Inter, Arial, sans-serif",
        "color": "#26332f",
        "size": 13,
    },
    "margin": {"l": 58, "r": 24, "t": 48, "b": 54},
    "hoverlabel": {"bgcolor": "#ffffff"},
}


def empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(**PLOT_LAYOUT)
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return figure


def trial_figure(rows: list[dict[str, Any]], color: str) -> go.Figure:
    if not rows:
        return empty_figure("Nessun trial con score")
    x = [row["trial"] for row in rows]
    y = [row["score"] for row in rows]
    best_so_far = list(np.maximum.accumulate(y))
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Trial",
            marker={
                "size": 10,
                "color": color,
                "line": {"width": 1, "color": "#ffffff"},
            },
            customdata=[
                [row["run_name"], row["optuna_state"]] for row in rows
            ],
            hovertemplate=(
                "Trial %{x}<br>Score %{y:.3f}<br>%{customdata[0]}"
                "<br>%{customdata[1]}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=best_so_far,
            mode="lines",
            name="Best-so-far",
            line={"color": "#1d2a26", "width": 2.5, "shape": "hv"},
            hovertemplate=(
                "Best fino al trial %{x}: %{y:.3f}<extra></extra>"
            ),
        )
    )
    layout = {
        **PLOT_LAYOUT,
        "margin": {"l": 58, "r": 24, "t": 76, "b": 54},
    }
    figure.update_layout(
        **layout,
        title={
            "text": "Evoluzione dell'obiettivo",
            "x": 0.02,
            "xanchor": "left",
        },
        legend={"orientation": "h", "x": 1, "xanchor": "right", "y": 1.12},
    )
    figure.update_xaxes(title="Numero trial", gridcolor="#e7ece9", dtick=2)
    figure.update_yaxes(title="sweep/mean_fast_return", gridcolor="#e7ece9")
    return figure


def distance_figure(rows: list[dict[str, Any]], top_k: int) -> go.Figure:
    labels, matrix, _ = distance_matrix(rows, top_k)
    if not labels:
        return empty_figure("Nessun candidato disponibile")
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            zmin=0,
            zmax=1,
            colorscale=[
                [0, "#f4f7f5"],
                [0.5, "#e8a328"],
                [1, "#b63b35"],
            ],
            colorbar={"title": "Distanza"},
            text=np.round(matrix, 2),
            texttemplate="%{text}",
            hovertemplate=(
                "%{x} vs %{y}<br>Distanza %{z:.3f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        **PLOT_LAYOUT, title="Distanza normalizzata tra i top-k"
    )
    figure.update_yaxes(autorange="reversed")
    return figure


def importance_figure(
    rows: list[dict[str, Any]],
    fanova_info: dict[str, Any],
    mode: str,
) -> go.Figure:
    if mode == "fanova":
        if fanova_info.get("error"):
            return empty_figure(f"fANOVA non disponibile: {fanova_info['error']}")
        entries = list(fanova_info.get("importances", {}).items())[:10]
        title = "Parameter importance fANOVA (Optuna)"
        x_title = "Importanza normalizzata (somma = 1)"
        color = "#1e9f79"
    else:
        entries = parameter_association(rows)[:10]
        title = "Associazione univariata con lo score"
        x_title = "Forza dell'associazione (0-1)"
        color = "#2f73c9"

    if not entries:
        return empty_figure("Parametri insufficienti")
    names, values = zip(*reversed(entries))
    figure = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=color,
            customdata=[
                fanova_info.get("completed_trials")
                if mode == "fanova"
                else len(rows)
            ] * len(values),
            hovertemplate=(
                "%{y}<br>Valore %{x:.3f}<br>"
                "Trial completati: %{customdata}<extra></extra>"
            ),
        )
    )
    figure.update_layout(**PLOT_LAYOUT, title=title)
    figure.update_xaxes(
        title=x_title,
        range=[0, 1],
        gridcolor="#e7ece9",
    )
    figure.update_yaxes(title=None)
    return figure


def parameter_figure(
    rows: list[dict[str, Any]], parameter: str | None, color: str
) -> go.Figure:
    pairs = [
        row
        for row in rows
        if parameter and parameter in row.get("params", {})
    ]
    if not pairs:
        return empty_figure("Seleziona un parametro")
    values = [row["params"][parameter] for row in pairs]
    numeric = all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in values
    )
    x = values if numeric else [str(value) for value in values]
    figure = go.Figure(
        go.Scatter(
            x=x,
            y=[row["score"] for row in pairs],
            mode="markers",
            marker={
                "size": 10,
                "color": color,
                "opacity": 0.8,
                "line": {"width": 1, "color": "#ffffff"},
            },
            customdata=[row["trial"] for row in pairs],
            hovertemplate=(
                "Trial %{customdata}<br>Valore %{x}<br>"
                "Score %{y:.3f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(**PLOT_LAYOUT, title=f"Score rispetto a {parameter}")
    figure.update_xaxes(title=parameter, gridcolor="#e7ece9")
    if numeric:
        numeric_values = [float(value) for value in values]
        if (
            min(numeric_values) > 0
            and max(numeric_values) / min(numeric_values) >= 100
        ):
            figure.update_xaxes(type="log")
    figure.update_yaxes(title="sweep/mean_fast_return", gridcolor="#e7ece9")
    return figure


def parallel_figure(rows: list[dict[str, Any]], top_k: int) -> go.Figure:
    ranked = sorted(rows, key=lambda row: row["score"], reverse=True)[:top_k]
    if len(ranked) < 2:
        return empty_figure("Servono almeno due candidati")
    dimensions: list[dict[str, Any]] = [
        {"label": "score", "values": [row["score"] for row in ranked]}
    ]
    for name in all_params(ranked):
        values = [row.get("params", {}).get(name) for row in ranked]
        if any(value is None for value in values):
            continue
        if all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        ):
            dimensions.append({"label": name, "values": values})
        else:
            labels = [str(value) for value in values]
            unique = sorted(set(labels))
            mapping = {label: index for index, label in enumerate(unique)}
            dimensions.append(
                {
                    "label": name,
                    "values": [mapping[label] for label in labels],
                    "tickvals": list(mapping.values()),
                    "ticktext": list(mapping.keys()),
                }
            )
    figure = go.Figure(
        go.Parcoords(
            line={
                "color": [row["score"] for row in ranked],
                "colorscale": [
                    [0, "#d7e2dc"],
                    [0.65, "#2f73c9"],
                    [1, "#e64f45"],
                ],
                "showscale": True,
                "colorbar": {"title": "Score"},
            },
            dimensions=dimensions,
            labelfont={"size": 11},
        )
    )
    layout = {**PLOT_LAYOUT, "margin": {"l": 48, "r": 70, "t": 78, "b": 34}}
    figure.update_layout(
        **layout,
        title="Configurazioni top-k in coordinate parallele",
    )
    return figure


def fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def metric_card(
    label: str,
    value: str,
    detail: str = "",
    tone: str = "neutral",
) -> html.Div:
    return html.Div(
        [
            html.Span(label, className="metric-label"),
            html.Strong(value),
            html.Small(detail),
        ],
        className=f"metric-card {tone}",
    )


def overview_cards(payload: dict[str, Any]) -> list[html.Div]:
    records = payload.get("records", [])
    cards = []
    for key, spec in ALGORITHMS.items():
        rows = scored(records, key)
        info = summary(rows)
        if not info:
            cards.append(metric_card(spec["label"], "Nessun dato"))
            continue
        cards.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(spec["label"]),
                            html.Span(
                                info["diagnosis"],
                                className=f"status {info['tone']}",
                            ),
                        ],
                        className="study-card-head",
                    ),
                    html.Strong(f"{info['best']['score']:.3f}"),
                    html.Small(
                        f"t{info['best']['trial']:03d} | "
                        f"gap {fmt(info['gap'])} | {len(rows)} scored"
                    ),
                ],
                className="study-card",
                style={"--accent": spec["color"]},
            )
        )
    return cards


initial_payload = load_cache()
app = Dash(
    __name__, title="Tuning thesis", suppress_callback_exceptions=True
)
server = app.server

algorithm_options = [
    {"label": spec["label"], "value": key}
    for key, spec in ALGORITHMS.items()
]

app.layout = html.Div(
    [
        dcc.Store(id="data-store", data=initial_payload),
        html.Header(
            [
                html.Div(
                    [
                        html.P("SUMO HUMAN FEEDBACK RL", className="eyebrow"),
                        html.H1("Tuning diagnostics"),
                        html.P(
                            "Convergenza, robustezza e struttura delle "
                            "configurazioni Optuna",
                            className="subtitle",
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Button(
                            "Sincronizza W&B",
                            id="sync-button",
                            n_clicks=0,
                        ),
                        html.Span(id="sync-status"),
                    ],
                    className="sync-block",
                ),
            ],
            className="page-header",
        ),
        html.Main(
            [
                html.Section(
                    [
                        html.Div(
                            id="overview-grid", className="overview-grid"
                        )
                    ],
                    className="overview-band",
                ),
                html.Section(
                    [
                        html.Div(
                            [
                                html.Label("Algoritmo"),
                                dcc.Dropdown(
                                    id="algorithm",
                                    options=algorithm_options,
                                    value="hybrid_bernoulli",
                                    clearable=False,
                                ),
                            ],
                            className="control wide",
                        ),
                        html.Div(
                            [
                                html.Label("Candidati confrontati"),
                                dcc.Slider(
                                    id="top-k",
                                    min=3,
                                    max=10,
                                    step=1,
                                    value=5,
                                    marks={
                                        3: "3",
                                        5: "5",
                                        8: "8",
                                        10: "10",
                                    },
                                ),
                            ],
                            className="control slider-control",
                        ),
                    ],
                    className="controls-band",
                ),
                html.Section(
                    [html.Div(id="study-metrics", className="metric-grid")],
                    className="metrics-band",
                ),
                dcc.Tabs(
                    id="tabs",
                    value="progress",
                    children=[
                        dcc.Tab(
                            label="Convergenza",
                            value="progress",
                            className="tab",
                            selected_className="tab tab--selected",
                            children=[
                                html.Section(
                                    [
                                        html.Div(
                                            [
                                                dcc.Graph(
                                                    id="trial-plot",
                                                    config=GRAPH_CONFIG,
                                                )
                                            ],
                                            className="plot-panel large",
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Label("Metodo"),
                                                        dcc.RadioItems(
                                                            id="importance-mode",
                                                            options=[
                                                                {
                                                                    "label": "fANOVA",
                                                                    "value": "fanova",
                                                                },
                                                                {
                                                                    "label": "Univariata",
                                                                    "value": "univariate",
                                                                },
                                                            ],
                                                            value="fanova",
                                                            inline=True,
                                                            className="segmented",
                                                        ),
                                                    ],
                                                    className="importance-toolbar",
                                                ),
                                                dcc.Graph(
                                                    id="importance-plot",
                                                    config=GRAPH_CONFIG,
                                                )
                                            ],
                                            className="plot-panel importance-panel",
                                        ),
                                    ],
                                    className="plot-grid",
                                )
                            ],
                        ),
                        dcc.Tab(
                            label="Configurazioni",
                            value="configs",
                            className="tab",
                            selected_className="tab tab--selected",
                            children=[
                                html.Section(
                                    [
                                        html.Div(
                                            [
                                                dcc.Graph(
                                                    id="parallel-plot",
                                                    config=GRAPH_CONFIG,
                                                )
                                            ],
                                            className="plot-panel full",
                                        )
                                    ],
                                    className="single-grid",
                                ),
                                html.Section(
                                    [
                                        html.Div(
                                            [
                                                dcc.Graph(
                                                    id="distance-plot",
                                                    config=GRAPH_CONFIG,
                                                )
                                            ],
                                            className="plot-panel",
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Parametro",
                                                    className="inline-label",
                                                ),
                                                dcc.Dropdown(
                                                    id="parameter",
                                                    clearable=False,
                                                ),
                                                dcc.Graph(
                                                    id="parameter-plot",
                                                    config=GRAPH_CONFIG,
                                                ),
                                            ],
                                            className=(
                                                "plot-panel parameter-panel"
                                            ),
                                        ),
                                    ],
                                    className="plot-grid equal",
                                ),
                            ],
                        ),
                        dcc.Tab(
                            label="Top candidati",
                            value="candidates",
                            className="tab",
                            selected_className="tab tab--selected",
                            children=[
                                html.Section(
                                    [
                                        dash_table.DataTable(
                                            id="candidate-table",
                                            sort_action="native",
                                            filter_action="native",
                                            page_size=10,
                                            style_table={
                                                "overflowX": "auto"
                                            },
                                        )
                                    ],
                                    className="table-band",
                                )
                            ],
                        ),
                    ],
                ),
                html.Footer(
                    [
                        html.Span(id="source-note"),
                        html.Span(
                            "Distanza Gower-like sui parametri variabili; "
                            "fANOVA sui trial completati; associazioni "
                            "univariate descrittive e non causali."
                        ),
                    ]
                ),
            ]
        ),
    ],
    className="app-shell",
)


@callback(
    Output("data-store", "data"),
    Output("sync-status", "children"),
    Input("sync-button", "n_clicks"),
    prevent_initial_call=True,
)
def refresh_data(_: int) -> tuple[dict[str, Any], str]:
    try:
        payload = sync_data()
        timestamp = (
            datetime.fromisoformat(payload["synced_at"])
            .astimezone()
            .strftime("%d/%m %H:%M")
        )
        return (
            payload,
            f"Aggiornato {timestamp} | {len(payload['records'])} run",
        )
    except Exception as error:
        return no_update, f"Errore: {error}"


@callback(
    Output("overview-grid", "children"),
    Output("source-note", "children"),
    Input("data-store", "data"),
)
def render_overview(
    payload: dict[str, Any],
) -> tuple[list[html.Div], str]:
    synced = payload.get("synced_at")
    when = (
        datetime.fromisoformat(synced).astimezone().strftime("%d/%m/%Y %H:%M")
        if synced
        else "mai"
    )
    source = (
        f"Fonte: W&B {payload.get('project')} + journal Optuna | sync {when}"
    )
    return overview_cards(payload), source


@callback(
    Output("parameter", "options"),
    Output("parameter", "value"),
    Input("algorithm", "value"),
    Input("data-store", "data"),
)
def set_parameters(
    algorithm: str, payload: dict[str, Any]
) -> tuple[list[dict[str, str]], str | None]:
    names = all_params(scored(payload.get("records", []), algorithm))
    return (
        [{"label": name, "value": name} for name in names],
        names[0] if names else None,
    )


@callback(
    Output("study-metrics", "children"),
    Output("trial-plot", "figure"),
    Output("importance-plot", "figure"),
    Output("parallel-plot", "figure"),
    Output("distance-plot", "figure"),
    Output("parameter-plot", "figure"),
    Output("candidate-table", "data"),
    Output("candidate-table", "columns"),
    Input("algorithm", "value"),
    Input("top-k", "value"),
    Input("parameter", "value"),
    Input("importance-mode", "value"),
    Input("data-store", "data"),
)
def render_study(
    algorithm: str,
    top_k: int,
    parameter: str | None,
    importance_mode: str,
    payload: dict[str, Any],
):
    rows = scored(payload.get("records", []), algorithm)
    color = ALGORITHMS[algorithm]["color"]
    info = summary(rows)
    if not info:
        metrics = [metric_card("Stato", "Nessun dato")]
    else:
        metrics = [
            metric_card(
                "Diagnosi",
                info["diagnosis"],
                "euristica descrittiva",
                info["tone"],
            ),
            metric_card(
                "Best",
                f"{info['best']['score']:.3f}",
                f"trial t{info['best']['trial']:03d}",
            ),
            metric_card(
                "Gap dal secondo", fmt(info["gap"], 3), "punti validation"
            ),
            metric_card(
                "Entro 1 punto",
                str(info["within_one"]),
                f"su {len(rows)} trial scored",
            ),
            metric_card(
                "Guadagno ultimi 10",
                fmt(info["late_gain"], 3),
                "rispetto al best precedente",
            ),
            metric_card(
                "Isolamento best",
                fmt(info["isolation"], 2),
                "rapporto rispetto agli altri top-5",
            ),
        ]
    table_data, columns = candidate_table(rows, int(top_k))
    return (
        metrics,
        trial_figure(rows, color),
        importance_figure(
            rows,
            payload.get("fanova", {}).get(algorithm, {}),
            importance_mode,
        ),
        parallel_figure(rows, int(top_k)),
        distance_figure(rows, int(top_k)),
        parameter_figure(rows, parameter, color),
        table_data,
        [{"name": name, "id": name} for name in columns],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--sync-only", action="store_true")
    args = parser.parse_args()
    if args.sync_only:
        payload = sync_data()
        print(f"Synced {len(payload['records'])} runs")
        return
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
