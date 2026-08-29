"""What the gradient and fusion metrics mean, written as formulas.

These are the contract between what a figure shows and what the code computed.
The alpha formulas come from `alpha_estimation.py`. The frozen-probe metrics
come from an earlier diagnostic branch that is no longer part of the package:
the runs that logged them are still on W&B, and this is what their axes mean.

Rendering goes through matplotlib mathtext rather than MathJax, so the selector
keeps no external dependencies. Mathtext covers a subset of LaTeX: no
environments, no `\\text{}` (use `\\mathrm{}`), and every line renders on its own.
"""
from __future__ import annotations

import io

K = r"K"

# Shared by every frozen-probe metric, shown once at the top of the panel
# rather than repeated in each entry.
PREAMBLE = [
    r"$K=32$ probe indipendenti agli stessi parametri $\theta_t$,"
    r" canale $c\in\{p,d\}$",
    r"$\hat g_c=\frac{1}{K}\sum_i g_c^{(i)}$"
    r"$\qquad$"
    r"$\bar g_c^{(i)}=g_c^{(i)}/\|g_c^{(i)}\|_2$"
    r"$\qquad$"
    r"$m_c=\frac{1}{K}\sum_i \bar g_c^{(i)}$",
]

# metric -> (title, [mathtext lines], prose note)
METRIC_FORMULAS: dict[str, tuple[str, list[str], str]] = {
    "reward/grad_probe_dir_var_pref": (
        "Directional variance (preferences)",
        [r"$\mathrm{CV}_p^2=\frac{1}{K-1}\sum_i\|\bar g_p^{(i)}-m_p\|_2^2"
         r"=\frac{K}{K-1}\left(1-\|m_p\|_2^2\right)$"],
        "Scatter of the directions alone, lengths divided out. 0 means every probe points the "
         "same way, 1 is what independent directions would give. The typical angle between a "
         "probe and the consensus is arccos of sqrt(1 - CV^2 (K-1)/K).",
    ),
    "reward/grad_probe_dir_var_demo": (
        "Directional variance (demonstrations)",
        [r"$\mathrm{CV}_d^2=\frac{1}{K-1}\sum_i\|\bar g_d^{(i)}-m_d\|_2^2$"],
        "The same quantity on the demonstration channel, and the denominator of the other term "
         "in alpha.",
    ),
    "reward/grad_probe_precond_dir_var_pref": (
        "Directional variance after Adam (preferences)",
        ['$\\mathrm{CV}_p^2$ computed on $u_p^{(i)}$ instead of $g_p^{(i)}$'],
        "Only exists in the two-Adam schemes. With one Adam on the already fused gradient "
         "there are no per-channel directions after Adam.",
    ),
    "reward/grad_probe_precond_dir_var_demo": (
        "Directional variance after Adam (demonstrations)",
        ['$\\mathrm{CV}_d^2$ computed on $u_d^{(i)}$ instead of $g_d^{(i)}$'],
        "In the two-Adam schemes this one, not the raw one, is what feeds alpha.",
    ),
    "reward/hybrid_alpha": (
        "Weight on the demonstrations",
        [r"$\alpha=\frac{\mathrm{CV}_p^2}{\mathrm{CV}_p^2+\mathrm{CV}_d^2}$"],
        'alpha weights the DEMONSTRATIONS, and rises when the preference channel scatters more. It is 1, demonstrations only, until the preference dataset holds five comparisons, because below that its dispersion cannot be estimated. Careful when comparing campaigns: in the earlier runs the two CV^2 were directional variances measured between probe batches and smoothed with an EMA; since then they come from the per-sample sampling variance and there is no EMA.',
    ),
    "reward/hybrid_alpha_active": (
        "α stimato o fissato",
        ['$1$ if $N_p\\ge 5$ and both dispersions are finite, $0$ otherwise'],
        'When this is 0 the alpha curve is flat at 1 by construction, not because the preference channel was judged unreliable: below five comparisons its dispersion cannot be estimated and the weight is pinned.',
    ),
    "reward/grad_probe_cosine_of_means": (
        "Cosine between the mean gradients",
        [r"$\cos=\frac{\langle \hat g_p,\hat g_d\rangle}"
         r"{\|\hat g_p\|_2\,\|\hat g_d\|_2}$"],
        "How aligned the systematic parts are. Averaging over K probes cuts the noise by "
         "sqrt(K). Independent directions in R^d would give about plus or minus 1/sqrt(d).",
    ),
    "reward/grad_probe_cosine": (
        "Coseno per campione",
        [r"$\overline{\cos}=\frac{1}{K}\sum_i\frac{\langle g_p^{(i)},g_d^{(i)}\rangle}"
         r"{\|g_p^{(i)}\|_2\,\|g_d^{(i)}\|_2}$"],
        "The mean of the cosines, not the cosine of the means. It carries the noise of every "
         "probe, so it sits closer to zero.",
    ),
    "reward/grad_probe_var_pref": (
        "Total variance (preferences)",
        [r"$\widehat V_p=\frac{1}{K-1}\sum_i\|g_p^{(i)}-\hat g_p\|_2^2$"],
        "In squared-gradient units, so unlike the directional variance it depends on the "
         "scale.",
    ),
    "reward/grad_probe_var_demo": (
        "Total variance (demonstrations)",
        [r"$\widehat V_d=\frac{1}{K-1}\sum_i\|g_d^{(i)}-\hat g_d\|_2^2$"],
        "",
    ),
    "reward/grad_probe_mean_sq_norm_pref": (
        "Norma quadratica del gradiente medio (preferenze)",
        [r"$\|\hat g_p\|_2^2\qquad$ stimatore gonfiato: "
         r"$\mathbb{E}\|\hat g_p\|_2^2=\|\bar g_p\|_2^2+\widehat V_p/K$"],
        "The V/K term is the noise left in the mean. When it is comparable to the estimate, "
         "the mean is noise.",
    ),
    "reward/grad_probe_mean_sq_norm_demo": (
        "Norma quadratica del gradiente medio (dimostrazioni)",
        [r"$\|\hat g_d\|_2^2$"],
        "",
    ),
    "reward/demo_2_expert_softmax_mass": (
        "Softmax mass on the experts",
        [r"$\sum_{i\in E} p_i,\quad p=\mathrm{softmax}\left(R_E\cup R_M\right)$"],
        "The share of the demo_2 partition carried by the expert trajectories. Near 1 the loss "
         "is saturated: the agent trajectories, the only source of variation, no longer count.",
    ),
    "reward_val/current_rollout/post_update/reward_std": (
        "Standard deviation of the reward the agent sees",
        ['$r_{\\mathrm{agent}}=(r-\\mu)/\\sigma$ with normalization on,$\\;r$ otherwise'],
        "With normalization on it is 1 by construction. Without it, it is the model's own "
         "scale, and the ratio between the two is the gain applied.",
    ),
    "replay_relabel_debug/delta_abs_mean": (
        "Scarto fra reward in buffer e ricalcolato",
        [r"$\frac{1}{N}\sum_j\left|r_{\mathrm{stored}}(j)-r_{\theta_t}(j)\right|$"],
        'mu and sigma are estimated on rollout t and applied to t+1, after the model has changed. This gap measures that lag.',
    ),
}

# --- alpha: per-sample sampling variance -------------------------------------
# N and B play different roles and are kept apart in the panel: confusing the
# two is what made the earlier estimate wrong.
_ALPHA_N_B = (
    '$N_c$ = samples available in the channel; $B_c=\\min(\\text{batch\\_size}_c, N_c)$ = the minibatch the optimizer uses'
)

METRIC_FORMULAS["alpha/V_pref"] = (
    "Process variance (preferences)",
    [r"$V_p=\frac{1}{N_p-1}\sum_{i=1}^{N_p}\bigl\|g_i^p-\bar g_p\bigr\|_2^2$",
     r"$g_i^p=(p_i-y_i)\,\nabla_\theta\Delta_i$ — gradiente del SINGOLO confronto"],
    'How much the gradient from a single piece of feedback scatters around the mean gradient. It does NOT depend on the budget: more samples make the estimate more precise, they do not change what is estimated. If it falls steadily with the budget, something is wrong.',
)
METRIC_FORMULAS["alpha/V_demo"] = (
    "Process variance (demonstrations)",
    [r"$V_d=\frac{1}{N_d-1}\sum_{i=1}^{N_d}\bigl\|g_i^d-\bar g_d\bigr\|_2^2$",
     r"$g_i^d=(w_{\text{last}}-1)\nabla R_i^E+\sum_j w_j\nabla R_j^M$, "
     r"$\;w=\mathrm{softmax}\bigl(\{R_j^M\}\cup\{R_i^E\}\bigr)$"],
    'The same quantity on the demonstration channel. `demo_2` does not decompose, so sample i is the loss of that one demonstration against the whole rollout, held frozen because it is not feedback.',
)
METRIC_FORMULAS["alpha/S_pref"] = (
    "Variance of the sample mean (preferences)",
    [r"$S_p=\dfrac{V_p}{B_p}$", _ALPHA_N_B],
    'The noise of the gradient the optimizer actually applies. It MUST fall as the budget grows: that is the sanity check on the estimator. The fall stops once the budget passes the minibatch, because B stops growing there.',
)
METRIC_FORMULAS["alpha/S_demo"] = (
    "Variance of the sample mean (demonstrations)",
    [r"$S_d=\dfrac{V_d}{B_d}$", _ALPHA_N_B],
    'At a large budget it settles at $S_p=V_p/256$ against $S_d=V_d/64$. The asymmetry between the two minibatches is deliberate: the preference gradient really is averaged over four times as many samples.',
)
METRIC_FORMULAS["alpha/cv2_pref"] = (
    "CV² (preferenze)",
    [r"$\mathrm{CV}_p^2=\dfrac{S_p}{\|\bar g_p\|_2^2}$"],
    'Made dimensionless. The two losses have different scales, and without dividing by the length of the mean gradient alpha would depend on `pref_temperature` and on the reward scale instead of on the statistics.',
)
METRIC_FORMULAS["alpha/cv2_demo"] = (
    "CV² (dimostrazioni)",
    [r"$\mathrm{CV}_d^2=\dfrac{S_d}{\|\bar g_d\|_2^2}$"],
    'The other ingredient of alpha. Only the ratio between the two CV^2 matters: a factor common to both channels does not move the weight.',
)
METRIC_FORMULAS["alpha/gradmean_norm_sq_pref"] = (
    "Squared norm of the mean gradient (preferences)",
    ['$\\|\\bar g_p\\|_2^2$, with $\\bar g_p=\\frac{1}{N_p}\\sum_i g_i^p$'],
    'The denominator of CV^2. When it collapses towards zero the channel has no systematic direction left and CV^2 blows up: this is what to look at when alpha sticks to an extreme.',
)
METRIC_FORMULAS["alpha/gradmean_norm_sq_demo"] = (
    "Squared norm of the mean gradient (demonstrations)",
    [r"$\|\bar g_d\|_2^2$"],
    'The same role on the demonstration channel.',
)
METRIC_FORMULAS["alpha/n_pref"] = (
    'Samples used to estimate V (preferences)',
    [r"$N_p=$ confronti raccolti finora"],
    'Grows through the run as the budget fills up. With no validation split it reaches exactly B; the old split would have stopped at 0.8 B.',
)
METRIC_FORMULAS["alpha/n_demo"] = (
    'Samples used to estimate V (demonstrations)',
    [r"$N_d=$ traiettorie esperte disponibili"],
    'Constant through the run: every demonstration is available from the start.',
)
METRIC_FORMULAS["alpha/batch_pref"] = (
    "Preference minibatch",
    [r"$B_p=\min(\text{batch\_size\_pref}, N_p)$"],
    'At a small budget it is clipped to the pool, so the minibatch is the whole dataset and the gradient is full-batch. That is the clipping, not a tuning choice.',
)
METRIC_FORMULAS["alpha/batch_demo"] = (
    "Demonstration minibatch",
    [r"$B_d=\min(\text{batch\_size\_expert}, N_d)$"],
    'The same clipping on the demonstration channel.',
)

METRIC_FORMULAS["reward/normalization_raw_std"] = (
    "Sigma grezzo del reward model",
    ['$\\sigma=\\mathrm{std}\\left(r_\\theta(\\tau)\\right)$ on the current rollout, before normalization'],
    'Logged only when normalization is on, where the gain applied to the agent is 1/sigma. Without normalization the key does not exist and the curve comes out empty.',
)
METRIC_FORMULAS["replay_relabel_debug/current_reward_std"] = (
    'Reward sigma in the replay buffer',
    [r"$\mathrm{std}\left(r_{\theta_t}(j)\right)$ sulle transizioni in buffer"],
    'The scale the critic actually regresses on, recomputed with the current model.',
)

# Aliases: these keys share a definition and a note.
METRIC_FORMULAS["reward_val/current_rollout/post_update/reward_mean"] = (
    "Mean reward the agent sees",
    [r"$r_{\mathrm{agente}}=(r-\mu)/\sigma$"],
    'Zero on the rollout mu was estimated from, not on the next one.',
)

ADAM_LINE = ('$u_c=\\mathrm{Adam}_c(g_c)=\\hat m_c/(\\sqrt{\\hat v_c}+\\epsilon)$, one state per channel')

# fusion -> (title, [mathtext lines])
FUSION_FORMULAS: dict[str, tuple[str, list[str]]] = {
    "norm_balance": ("norm_balance (baseline)", [
        r"$s=\min\left(w\frac{\|g_p\|_2}{\|g_d\|_2+\epsilon},\,100\right)$",
        r"$\theta\leftarrow\theta-\eta\,\mathrm{Adam}(g_p+s\,g_d)$",
    ]),
    "alpha_norm_single_adam": ("one Adam on the fused gradient", [
        r"$g^{\mathrm{fin}}=(1-\alpha)\,\bar g_p+\alpha\,\bar g_d$",
        r"$\theta\leftarrow\theta-\eta\,\mathrm{Adam}(g^{\mathrm{fin}})$",
    ]),
    "dual_adam_alpha": ("one Adam per channel", [
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
    "dual_adam_alpha_unit_nobudget": ("two Adams, alpha on unit directions", [
        r"$\theta\leftarrow\theta-\eta\left[(1-\alpha)\frac{u_p}{\|u_p\|_2}"
        r"+\alpha\frac{u_d}{\|u_d\|_2}\right]\qquad(B=1)$",
    ]),
}


def blocks(metric: str | None, fusions=()) -> list[dict]:
    """The blocks to show: the metric definition, plus the selected fusions."""
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
        note = ("alpha weights the demonstrations; u_p and u_d are the Adam updates, one state "
                 "per channel." if any(f != "norm_balance" for f in known) else "")
        lines: list[str] = []
        if any(f != "norm_balance" and f != "alpha_norm_single_adam" for f in known):
            lines.append(ADAM_LINE)
        for f in known:
            title, body = FUSION_FORMULAS[f]
            lines.append(f"__{title}__")
            lines.extend(body)
        out.append({"title": 'Fusion schemes in the selection',
                    "lines": lines, "note": note})
    return out


def render_png(blocks_: list[dict], width: float = 4.6, dpi: int = 300) -> bytes:
    """The same blocks as a raster image, to sit beside the exported figure.

    Drawn by the same function as the SVG, so the panel and the image agree.
    """
    return _render(blocks_, width, fmt="png", dpi=dpi)


def render_svg(blocks_: list[dict], width: float = 4.6) -> str:
    """The blocks as SVG, through mathtext, so the page needs no JavaScript."""
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

    # Estimated height per line: notes wrap, so they count for more than a
    # formula does.
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
        # A fraction or a sum reaches above and below the baseline: without the
        # extra room the numerator lands on the line before.
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
