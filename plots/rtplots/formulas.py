"""Le definizioni matematiche delle metriche dei gradienti e delle fusioni.

Trascritte da `human_feedback_rl/algorithms/hybrid/gradient_statistics.py` e da
`hybrid_algorithm.py:_alpha_weight/_fusion_components`, non dalla memoria: sono
il contratto fra quello che il grafico mostra e quello che il codice calcola, e
se divergono e' la formula qui a essere sbagliata.

Il rendering passa da mathtext di matplotlib (gia' una dipendenza del toolkit)
invece che da MathJax/KaTeX, cosi' il selettore resta senza dipendenze
esterne come il resto della pagina. Mathtext copre un sottoinsieme di LaTeX:
niente ambienti (`align`, `cases`), niente `\\text{}` — si usa `\\mathrm{}` — e
ogni riga va resa per conto suo.
"""
from __future__ import annotations

import io

K = r"K"

# Valgono per tutte le metriche del frozen probe, mostrate una volta sola in
# testa al pannello invece che ripetute in ogni voce.
PREAMBLE = [
    r"$K=32$ probe indipendenti agli stessi parametri $\theta_t$,"
    r" canale $c\in\{p,d\}$",
    r"$\hat g_c=\frac{1}{K}\sum_i g_c^{(i)}$"
    r"$\qquad$"
    r"$\bar g_c^{(i)}=g_c^{(i)}/\|g_c^{(i)}\|_2$"
    r"$\qquad$"
    r"$m_c=\frac{1}{K}\sum_i \bar g_c^{(i)}$",
]

# metrica -> (titolo, [righe mathtext], nota in prosa)
METRIC_FORMULAS: dict[str, tuple[str, list[str], str]] = {
    "reward/grad_probe_dir_var_pref": (
        "Varianza direzionale (preferenze)",
        [r"$\mathrm{CV}_p^2=\frac{1}{K-1}\sum_i\|\bar g_p^{(i)}-m_p\|_2^2"
         r"=\frac{K}{K-1}\left(1-\|m_p\|_2^2\right)$"],
        "Dispersione delle sole direzioni, le lunghezze divise via. 0 = tutti i "
        "probe puntano uguale; 1 = direzioni indipendenti (riferimento del caso). "
        "L'angolo tipico fra un probe e il consenso è arccos√(1−CV²·(K−1)/K).",
    ),
    "reward/grad_probe_dir_var_demo": (
        "Varianza direzionale (dimostrazioni)",
        [r"$\mathrm{CV}_d^2=\frac{1}{K-1}\sum_i\|\bar g_d^{(i)}-m_d\|_2^2$"],
        "Stessa quantità sul canale delle dimostrazioni. È il denominatore "
        "dell'altro termine di α.",
    ),
    "reward/grad_probe_precond_dir_var_pref": (
        "Varianza direzionale post-Adam (preferenze)",
        [r"$\mathrm{CV}_p^2$ calcolata su $u_p^{(i)}$ invece che su $g_p^{(i)}$"],
        "Esiste solo negli schemi a due Adam: con un Adam solo sul gradiente già "
        "fuso non ci sono direzioni post-Adam per canale.",
    ),
    "reward/grad_probe_precond_dir_var_demo": (
        "Varianza direzionale post-Adam (dimostrazioni)",
        [r"$\mathrm{CV}_d^2$ calcolata su $u_d^{(i)}$ invece che su $g_d^{(i)}$"],
        "È questa, non la grezza, l'ingresso di α negli schemi a due Adam.",
    ),
    "reward/hybrid_alpha": (
        "Peso delle dimostrazioni",
        [r"$\alpha=\frac{\mathrm{CV}_p^2}{\mathrm{CV}_p^2+\mathrm{CV}_d^2}$"],
        "α pesa le DIMOSTRAZIONI e cresce quando il canale preferenze è più "
        "disperso. Vale 1 (solo dimostrazioni) finché il dataset di preferenze ha "
        "meno di 5 confronti, cioè finché la sua dispersione non è stimabile. "
        "ATTENZIONE al confronto fra campagne: nelle run precedenti ad agosto 2026 "
        "le due CV² erano varianze direzionali misurate FRA probe batch e lisciate "
        "con una EMA a β=0.9; da lì in poi vengono dalla varianza di campionamento "
        "per campione (vedi S_p, S_d) e non c'è più alcuna EMA.",
    ),
    "reward/hybrid_alpha_active": (
        "α stimato o fissato",
        [r"$1$ se $N_p\ge 5$ e le due dispersioni sono finite, $0$ altrimenti"],
        "Quando vale 0 la curva di α è piatta a 1 per costruzione, non perché il "
        "canale preferenze sia stato giudicato inaffidabile: sotto cinque confronti "
        "la sua dispersione non è stimabile e il peso è fissato d'ufficio.",
    ),
    "reward/grad_probe_cosine_of_means": (
        "Coseno fra i gradienti medi",
        [r"$\cos=\frac{\langle \hat g_p,\hat g_d\rangle}"
         r"{\|\hat g_p\|_2\,\|\hat g_d\|_2}$"],
        "Allineamento della parte sistematica: mediare su K probe abbatte il "
        "rumore di √K. Fra direzioni indipendenti in R^d ci si aspetta ±1/√d.",
    ),
    "reward/grad_probe_cosine": (
        "Coseno per campione",
        [r"$\overline{\cos}=\frac{1}{K}\sum_i\frac{\langle g_p^{(i)},g_d^{(i)}\rangle}"
         r"{\|g_p^{(i)}\|_2\,\|g_d^{(i)}\|_2}$"],
        "Media dei coseni, non coseno delle medie: include il rumore di ogni "
        "singolo probe, quindi è attenuato verso 0 rispetto all'altro.",
    ),
    "reward/grad_probe_var_pref": (
        "Varianza totale (preferenze)",
        [r"$\widehat V_p=\frac{1}{K-1}\sum_i\|g_p^{(i)}-\hat g_p\|_2^2$"],
        "In unità di gradiente al quadrato: dipende dalla scala, a differenza "
        "della varianza direzionale.",
    ),
    "reward/grad_probe_var_demo": (
        "Varianza totale (dimostrazioni)",
        [r"$\widehat V_d=\frac{1}{K-1}\sum_i\|g_d^{(i)}-\hat g_d\|_2^2$"],
        "",
    ),
    "reward/grad_probe_mean_sq_norm_pref": (
        "Norma quadratica del gradiente medio (preferenze)",
        [r"$\|\hat g_p\|_2^2\qquad$ stimatore gonfiato: "
         r"$\mathbb{E}\|\hat g_p\|_2^2=\|\bar g_p\|_2^2+\widehat V_p/K$"],
        "Il termine V/K è il rumore residuo della media: se è confrontabile con "
        "il valore stimato, la 'media' è rumore.",
    ),
    "reward/grad_probe_mean_sq_norm_demo": (
        "Norma quadratica del gradiente medio (dimostrazioni)",
        [r"$\|\hat g_d\|_2^2$"],
        "",
    ),
    "reward/demo_2_expert_softmax_mass": (
        "Massa softmax sugli esperti",
        [r"$\sum_{i\in E} p_i,\quad p=\mathrm{softmax}\left(R_E\cup R_M\right)$"],
        "Quota della funzione di partizione della demo_2 sostenuta dalle "
        "traiettorie esperte. Vicino a 1 la loss è satura: le traiettorie "
        "dell'agente, unica sorgente di variabilità, non pesano più.",
    ),
    "reward_val/current_rollout/post_update/reward_std": (
        "Deviazione standard del reward visto dall'agente",
        [r"$r_{\mathrm{agente}}=(r-\mu)/\sigma$ con normalizzazione attiva,"
         r"$\;r$ altrimenti"],
        "Con la normalizzazione attiva vale 1 per costruzione; senza, è la scala "
        "naturale del modello. Il rapporto fra le due è il guadagno imposto.",
    ),
    "replay_relabel_debug/delta_abs_mean": (
        "Scarto fra reward in buffer e ricalcolato",
        [r"$\frac{1}{N}\sum_j\left|r_{\mathrm{stored}}(j)-r_{\theta_t}(j)\right|$"],
        "μ e σ sono stimati sul rollout t e applicati al t+1, dopo che il modello "
        "è cambiato: questo scarto misura quel ritardo.",
    ),
}

# --- stima di α: varianza di campionamento per campione ----------------------
# N e B hanno ruoli diversi e vanno tenuti distinti nel pannello, perché è
# proprio la loro confusione che rendeva sbagliata la stima precedente.
_ALPHA_N_B = (
    r"$N_c$ = campioni disponibili nel canale; "
    r"$B_c=\min(\text{batch\_size}_c, N_c)$ = minibatch usato dall'ottimizzatore"
)

METRIC_FORMULAS["alpha/V_pref"] = (
    "Varianza del processo (preferenze)",
    [r"$V_p=\frac{1}{N_p-1}\sum_{i=1}^{N_p}\bigl\|g_i^p-\bar g_p\bigr\|_2^2$",
     r"$g_i^p=(p_i-y_i)\,\nabla_\theta\Delta_i$ — gradiente del SINGOLO confronto"],
    "Quanto disperde il gradiente indotto da un singolo feedback attorno al "
    "gradiente medio. NON dipende dal budget: più campioni migliorano la "
    "precisione della stima, non il valore stimato. Se cala sistematicamente col "
    "budget c'è qualcosa che non torna.",
)
METRIC_FORMULAS["alpha/V_demo"] = (
    "Varianza del processo (dimostrazioni)",
    [r"$V_d=\frac{1}{N_d-1}\sum_{i=1}^{N_d}\bigl\|g_i^d-\bar g_d\bigr\|_2^2$",
     r"$g_i^d=(w_{\text{last}}-1)\nabla R_i^E+\sum_j w_j\nabla R_j^M$, "
     r"$\;w=\mathrm{softmax}\bigl(\{R_j^M\}\cup\{R_i^E\}\bigr)$"],
    "Stessa quantità sul canale dimostrazioni. `demo_2` non si decompone, quindi "
    "il campione i-esimo è definito come la loss che si vedrebbe con quella sola "
    "dimostrazione più tutto il rollout, tenuto congelato perché non è feedback.",
)
METRIC_FORMULAS["alpha/S_pref"] = (
    "Varianza della media campionaria (preferenze)",
    [r"$S_p=\dfrac{V_p}{B_p}$", _ALPHA_N_B],
    "È il rumore del gradiente che l'ottimizzatore applica davvero. DEVE calare "
    "al crescere del budget: è il sanity check dello stimatore. Il calo si ferma "
    "dove il budget supera il minibatch, perché lì B smette di crescere.",
)
METRIC_FORMULAS["alpha/S_demo"] = (
    "Varianza della media campionaria (dimostrazioni)",
    [r"$S_d=\dfrac{V_d}{B_d}$", _ALPHA_N_B],
    "A budget grande resta $S_p=V_p/256$ contro $S_d=V_d/64$: l'asimmetria fra i "
    "due minibatch è voluta, perché il gradiente delle preferenze è davvero "
    "mediato su quattro volte più campioni.",
)
METRIC_FORMULAS["alpha/cv2_pref"] = (
    "CV² (preferenze)",
    [r"$\mathrm{CV}_p^2=\dfrac{S_p}{\|\bar g_p\|_2^2}$"],
    "Reso adimensionale: le due loss hanno scale diverse, e senza dividere per la "
    "lunghezza del gradiente medio α dipenderebbe da `pref_temperature` e dalla "
    "scala del reward invece che dalla statistica.",
)
METRIC_FORMULAS["alpha/cv2_demo"] = (
    "CV² (dimostrazioni)",
    [r"$\mathrm{CV}_d^2=\dfrac{S_d}{\|\bar g_d\|_2^2}$"],
    "L'altro ingrediente di α. Il rapporto fra i due CV² è tutto ciò che conta: "
    "un fattore comune ai due canali non sposta il peso.",
)
METRIC_FORMULAS["alpha/gradmean_norm_sq_pref"] = (
    "Norma quadratica del gradiente medio (preferenze)",
    [r"$\|\bar g_p\|_2^2$, con $\bar g_p=\frac{1}{N_p}\sum_i g_i^p$"],
    "Denominatore di CV². Quando collassa verso zero il canale non ha più una "
    "direzione sistematica e CV² esplode: è il segnale da guardare se α si "
    "incolla a un estremo.",
)
METRIC_FORMULAS["alpha/gradmean_norm_sq_demo"] = (
    "Norma quadratica del gradiente medio (dimostrazioni)",
    [r"$\|\bar g_d\|_2^2$"],
    "Stesso ruolo sul canale dimostrazioni.",
)
METRIC_FORMULAS["alpha/n_pref"] = (
    "Campioni usati per stimare V (preferenze)",
    [r"$N_p=$ confronti raccolti finora"],
    "Cresce lungo la run col riempirsi del budget. Senza validation set tende a B "
    "esatto; con lo split vecchio si sarebbe fermato a 0.8·B.",
)
METRIC_FORMULAS["alpha/n_demo"] = (
    "Campioni usati per stimare V (dimostrazioni)",
    [r"$N_d=$ traiettorie esperte disponibili"],
    "Costante lungo la run: le dimostrazioni sono tutte disponibili dall'inizio.",
)
METRIC_FORMULAS["alpha/batch_pref"] = (
    "Minibatch delle preferenze",
    [r"$B_p=\min(\text{batch\_size\_pref}, N_p)$"],
    "A budget piccolo è troncato al pool, quindi il minibatch coincide col "
    "dataset intero e il gradiente è full-batch: non è una scelta del tuning, è "
    "un effetto del troncamento.",
)
METRIC_FORMULAS["alpha/batch_demo"] = (
    "Minibatch delle dimostrazioni",
    [r"$B_d=\min(\text{batch\_size\_expert}, N_d)$"],
    "Stesso troncamento sul canale dimostrazioni.",
)

METRIC_FORMULAS["reward/normalization_raw_std"] = (
    "Sigma grezzo del reward model",
    [r"$\sigma=\mathrm{std}\left(r_\theta(\tau)\right)$ sul rollout corrente,"
     r" prima della normalizzazione"],
    "Loggata solo quando la normalizzazione e' attiva: il guadagno imposto "
    "all'agente e' 1/sigma. Nel braccio senza normalizzazione la chiave non "
    "esiste, quindi la curva risulta vuota.",
)
METRIC_FORMULAS["replay_relabel_debug/current_reward_std"] = (
    "Sigma del reward nel replay buffer",
    [r"$\mathrm{std}\left(r_{\theta_t}(j)\right)$ sulle transizioni in buffer"],
    "E' la scala su cui il critico fa davvero regressione, ricalcolata col "
    "modello corrente.",
)

# Alias: le due chiavi condividono definizione e nota.
METRIC_FORMULAS["reward_val/current_rollout/post_update/reward_mean"] = (
    "Media del reward visto dall'agente",
    [r"$r_{\mathrm{agente}}=(r-\mu)/\sigma$"],
    "Vale 0 sul rollout su cui μ è stato stimato, non su quello successivo.",
)

ADAM_LINE = (r"$u_c=\mathrm{Adam}_c(g_c)=\hat m_c/(\sqrt{\hat v_c}+\epsilon)$,"
             r" uno stato per canale")

# fusione -> (titolo, [righe mathtext])
FUSION_FORMULAS: dict[str, tuple[str, list[str]]] = {
    "norm_balance": ("norm_balance (baseline)", [
        r"$s=\min\left(w\frac{\|g_p\|_2}{\|g_d\|_2+\epsilon},\,100\right)$",
        r"$\theta\leftarrow\theta-\eta\,\mathrm{Adam}(g_p+s\,g_d)$",
    ]),
    "alpha_norm_single_adam": ("prova 1 — un Adam sul gradiente fuso", [
        r"$g^{\mathrm{fin}}=(1-\alpha)\,\bar g_p+\alpha\,\bar g_d$",
        r"$\theta\leftarrow\theta-\eta\,\mathrm{Adam}(g^{\mathrm{fin}})$",
    ]),
    "dual_adam_alpha": ("prova 2 — un Adam per canale", [
        r"$\theta\leftarrow\theta-\eta\left[(1-\alpha)\,u_p+\alpha\,u_d\right]$",
    ]),
    "dual_adam_sum": ("due Adam, somma", [
        r"$\theta\leftarrow\theta-\eta\left[u_p+u_d\right]$",
    ]),
    "dual_adam_alpha_unit": ("due Adam, alpha + budget", [
        r"$B=\|u_p\|_2+\|u_d\|_2$",
        r"$\theta\leftarrow\theta-\eta\,B\left[(1-\alpha)\frac{u_p}{\|u_p\|_2}"
        r"+\alpha\frac{u_d}{\|u_d\|_2}\right]$",
    ]),
    "dual_adam_alpha_unit_nobudget": ("due Adam, alpha su direzioni unitarie", [
        r"$\theta\leftarrow\theta-\eta\left[(1-\alpha)\frac{u_p}{\|u_p\|_2}"
        r"+\alpha\frac{u_d}{\|u_d\|_2}\right]\qquad(B=1)$",
    ]),
}


def blocks(metric: str | None, fusions=()) -> list[dict]:
    """Blocchi da mostrare: definizione della metrica + fusioni in selezione."""
    out: list[dict] = []
    entry = METRIC_FORMULAS.get(metric or "")
    if entry:
        title, lines, note = entry
        needs_probe = "grad_probe" in (metric or "") or metric == "reward/hybrid_alpha"
        out.append({"title": title,
                    "lines": (PREAMBLE if needs_probe else []) + list(lines),
                    "note": note})
    known = [f for f in dict.fromkeys(fusions) if f in FUSION_FORMULAS]
    if known:
        note = ("α pesa le dimostrazioni; u_p e u_d sono gli update di Adam, "
                "uno stato per canale." if any(f != "norm_balance" for f in known) else "")
        lines: list[str] = []
        if any(f != "norm_balance" and f != "alpha_norm_single_adam" for f in known):
            lines.append(ADAM_LINE)
        for f in known:
            title, body = FUSION_FORMULAS[f]
            lines.append(f"__{title}__")
            lines.extend(body)
        out.append({"title": "Schemi di fusione nella selezione",
                    "lines": lines, "note": note})
    return out


def render_png(blocks_: list[dict], width: float = 4.6, dpi: int = 300) -> bytes:
    """Gli stessi blocchi in raster, per affiancarli alla figura esportata.

    Stessa funzione di disegno dell'SVG: quello che vedi nel pannello e' quello
    che finisce nell'immagine.
    """
    return _render(blocks_, width, fmt="png", dpi=dpi)


def render_svg(blocks_: list[dict], width: float = 4.6) -> str:
    """I blocchi come SVG, via mathtext (nessuna dipendenza JS nella pagina)."""
    return _render(blocks_, width, fmt="svg")


def _render(blocks_: list[dict], width: float, fmt: str, dpi: int = 100):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [("title", b["title"]) for b in blocks_ for _ in (0,)]
    lines: list[tuple[str, str]] = []
    for i, b in enumerate(blocks_):
        if i:
            lines.append(("gap", ""))
        lines.append(("title", b["title"]))
        for text in b["lines"]:
            lines.append(("sub", text[2:-2]) if text.startswith("__") else ("math", text))
        if b["note"]:
            lines.append(("note", b["note"]))
    del rows

    # Altezza stimata per riga: le note vanno mandate a capo, quindi contano di
    # piu' di una formula.
    import textwrap
    laid: list[tuple[str, str]] = []
    for kind, text in lines:
        if kind == "note":
            for chunk in textwrap.wrap(text, 62) or [""]:
                laid.append(("note", chunk))
        else:
            laid.append((kind, text))
    base = {"title": 0.30, "sub": 0.24, "math": 0.32, "note": 0.19, "gap": 0.18}

    def height(kind: str, text: str) -> float:
        # Una frazione o una sommatoria occupano sopra e sotto la linea di base:
        # senza spazio in piu' il numeratore finisce sulla riga precedente.
        extra = 0.16 if kind == "math" and ("frac" in text or "sum" in text) else 0.0
        return base[kind] + extra

    laid_h = [height(k, t) for k, t in laid]
    total = sum(laid_h) + 0.15

    fig = plt.figure(figsize=(width, total))
    y = 1.0
    for (kind, text), raw_h in zip(laid, laid_h):
        step = raw_h / total
        y -= step
        if kind == "gap":
            continue
        style = {
            "title": dict(fontsize=9.5, fontweight="semibold", color="#1a1a1a"),
            "sub": dict(fontsize=8.5, color="#555b63", style="italic"),
            "math": dict(fontsize=10.5, color="#1a1a1a"),
            "note": dict(fontsize=8, color="#555b63"),
        }[kind]
        indent = 0.04 if kind in ("math", "sub") else 0.0
        fig.text(0.02 + indent, y + step * 0.15, text, ha="left", va="baseline", **style)
    if fmt == "svg":
        buf = io.StringIO()
        fig.savefig(buf, format="svg", transparent=True,
                    bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        svg = buf.getvalue()
        return svg[svg.index("<svg"):]
    raw = io.BytesIO()
    fig.savefig(raw, format=fmt, dpi=dpi, facecolor="white",
                bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return raw.getvalue()
