import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau


# ── Normalization ─────────────────────────────────────────────────────────────

def _normalization_params(all_true, all_pred, robust=False):
    """Return (pred_center, pred_scale, true_center, true_scale) for pred→true rescaling."""
    if robust:
        pred_center = np.median(all_pred)
        pred_scale  = (np.percentile(all_pred, 75) - np.percentile(all_pred, 25)) + 1e-8
        true_center = np.median(all_true)
        true_scale  = (np.percentile(all_true, 75) - np.percentile(all_true, 25)) + 1e-8
    else:
        pred_center, pred_scale = all_pred.mean(), all_pred.std() + 1e-8
        true_center, true_scale = all_true.mean(), all_true.std() + 1e-8
    return pred_center, pred_scale, true_center, true_scale

def _normalize_pred_to_true(true_ret, pred_ret, robust=False):
    """Rescale pred_ret into the same location/scale as true_ret."""
    pc, ps, tc, ts = _normalization_params(true_ret, pred_ret, robust)
    return (pred_ret - pc) / ps * ts + tc


# ── Segment extraction ────────────────────────────────────────────────────────

def _extract_segments(trajs, segment_length):
    """
    Non-overlapping segments extracted from the END of each trajectory,
    so the final segment is always present. None = full episode.
    Returned in chronological order.
    """
    segments = []
    for traj in trajs:
        traj_list = list(traj)
        n = len(traj_list)
        if segment_length is None:
            segments.append(traj_list)
        else:
            traj_segs = []
            end = n
            while end >= segment_length:
                start = end - segment_length
                traj_segs.append(traj_list[start:end])
                end = start
            if end > 0:  # leftover shorter than segment_length: take [0:segment_length], overlapping
                traj_segs.append(traj_list[:segment_length])
            segments.extend(reversed(traj_segs))
    return segments


# ── Scoring ───────────────────────────────────────────────────────────────────

# next_status: 7-dim one-hot [arrived, collided, off_road, timeout, running, teleported, removed_unknown]
_STATUS_LABELS = {
    0: "arrived", 1: "collision", 2: "off_road", 3: "timeout",
    4: "running", 5: "teleported", 6: "removed",
}
_STATUS_COLORS = {
    0: "steelblue", 1: "orange", 2: "crimson", 3: "mediumpurple",
    4: "gray",      5: "gold",   6: "saddlebrown",
}

def _score_segments(segments, rm):
    """Return (true_returns, pred_returns, terminal_status) as np.ndarray."""
    true_returns, pred_returns, statuses = [], [], []
    for seg in segments:
        obs  = np.array([t.observation for t in seg], dtype=np.float32)
        acts = np.array([t.action      for t in seg], dtype=np.float32)
        ns   = np.array([t.next_status for t in seg], dtype=np.float32)
        dn   = np.array([float(t.done) for t in seg], dtype=np.float32)
        pred_mean, _ = rm.predict_mean_std(obs, acts, ns, dn)
        true_returns.append(np.mean([t.true_reward for t in seg]))
        pred_returns.append(float(pred_mean.mean()))
        statuses.append(int(np.argmax(seg[-1].next_status)))
    return np.array(true_returns), np.array(pred_returns), np.array(statuses, dtype=int)


def _score_and_normalize(segments, rm, normalize=False, robust=False):
    """Score segments, optionally normalize pred to true scale, return correlations.

    Returns: (true_ret, pred_ret, term_status, pearson_r, spearman_rho, kendall_tau)
    """
    true_ret, pred_ret, term_status = _score_segments(segments, rm)
    if normalize:
        pred_ret = _normalize_pred_to_true(true_ret, pred_ret, robust)
    pr, _ = pearsonr(true_ret, pred_ret)
    sr, _ = spearmanr(true_ret, pred_ret)
    kt, _ = kendalltau(true_ret, pred_ret)
    return true_ret, pred_ret, term_status, pr, sr, kt


def _global_limits(scores, pad_frac=0.04):
    """Padded (xlim, ylim) from an iterable of (true_ret, pred_ret, ...) tuples.

    Use to share the same axis scale across a grid of scatter plots.
    """
    all_true = np.concatenate([s[0] for s in scores])
    all_pred = np.concatenate([s[1] for s in scores])
    pad = ((all_true.max() - all_true.min()) + (all_pred.max() - all_pred.min())) * (pad_frac / 2)
    return (all_true.min() - pad, all_true.max() + pad), (all_pred.min() - pad, all_pred.max() + pad)


# ── Per-axes scatter drawing ──────────────────────────────────────────────────

def _draw_scatter_ax(ax, true_ret, pred_ret, term_status,
                     xlim=None, ylim=None, title=None,
                     xlabel="True return", ylabel="Pred return",
                     s=15, alpha=0.7):
    """Draw a scatter of true vs pred returns into *ax*.

    xlim/ylim: provide for shared-scale grids; otherwise auto-scaled from data.
    """
    for status_id, color in _STATUS_COLORS.items():
        mask = term_status == status_id
        if not mask.any():
            continue
        ax.scatter(true_ret[mask], pred_ret[mask],
                   alpha=alpha, s=s, edgecolors="none", color=color,
                   label=_STATUS_LABELS.get(status_id))

    x_lo, x_hi = xlim if xlim is not None else ax.get_xlim()
    y_lo, y_hi = ylim if ylim is not None else ax.get_ylim()
    lo, hi = min(x_lo, y_lo), max(x_hi, y_hi)
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y=x")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    if title is not None:
        ax.set_title(title, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=6)


# ── Scatter plots ─────────────────────────────────────────────────────────────

def plot_scatter(trajs, rm, segment_length=None, normalize=False, robust=False):
    segments = _extract_segments(trajs, segment_length)
    plot_scatter_from_segments(segments, rm, normalize, robust)


def plot_scatter_from_segments(segments, rm, normalize=False, robust=False, title=None, ax=None):
    """Scatter of true return vs predicted return for pre-extracted segments.

    If *ax* is given, draws into it (caller is responsible for plt.show).
    Otherwise creates and shows a standalone figure.
    """
    if not segments:
        print("No segments provided.")
        return None, None

    true_ret, pred_ret, term_status, pr, sr, kt = _score_and_normalize(
        segments, rm, normalize, robust
    )
    align_tag  = f" ({'robust' if robust else 'z-score'} aligned)" if normalize else ""
    plot_title = title or f"Scatter  (L={len(segments[0])}, n={len(segments)})"
    full_title = f"{plot_title}\nPearson r={pr:.3f}   Spearman ρ={sr:.3f}   Kendall τ={kt:.3f}"

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 5))

    _draw_scatter_ax(ax, true_ret, pred_ret, term_status,
                     title=full_title,
                     xlabel="True return", ylabel=f"Predicted return{align_tag}",
                     s=25, alpha=0.75)
    ax.legend()

    if standalone:
        plt.tight_layout()
        plt.show()
        print(f"Pearson={pr:.3f}  Spearman={sr:.3f}  Kendall={kt:.3f}  n={len(segments)}")

    return ax.get_figure(), ax


def plot_scatter_all_models(trajs, reward_models, seg_lens, normalize=False, robust=False, rm_labels=None):
    """Grid scatter: rows = reward models, cols = seg_len in [1, 20, None].

    All subplots share the same x/y axis limits.
    """
    SEG_LENS   = seg_lens
    COL_LABELS = [f"seg_len={seg_len}" for seg_len in SEG_LENS]

    n_rows = len(reward_models)
    n_cols = len(SEG_LENS)

    all_segments = [_extract_segments(trajs, sl) for sl in SEG_LENS]

    scores = {}
    for row_idx, rm in enumerate(reward_models):
        for col_idx, segs in enumerate(all_segments):
            scores[(row_idx, col_idx)] = _score_and_normalize(segs, rm, normalize, robust)

    xlim, ylim = _global_limits(scores.values())

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 4), squeeze=False)

    for row_idx in range(n_rows):
        row_label = rm_labels[row_idx] if rm_labels else f"Model {row_idx}"
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            true_ret, pred_ret, term_status, pr, sr, _ = scores[(row_idx, col_idx)]
            n_seg  = len(all_segments[col_idx])
            stats  = f"r={pr:.2f}  ρ={sr:.2f}  n={n_seg}"
            title  = f"{COL_LABELS[col_idx]}\n{stats}" if row_idx == 0 else stats
            ylabel = f"{row_label}\nPred return" if col_idx == 0 else "Pred return"

            _draw_scatter_ax(ax, true_ret, pred_ret, term_status,
                             xlim=xlim, ylim=ylim,
                             title=title, xlabel="True return", ylabel=ylabel)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
    align_tag = f" ({'robust' if robust else 'z-score'} aligned)" if normalize else " (raw)"
    plt.suptitle(f"Scatter — all models × seg_len{align_tag}", fontsize=11)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.show()


def plot_scatter_checkpoints(run_dir, segments, load_fn, obs_space, act_space,
                             n_cols=4, normalize=False, robust=False):
    """Grid of scatter plots for every checkpoint under *run_dir*, shared axis scale."""
    from pathlib import Path

    run_dir = Path(run_dir)
    checkpoints = sorted(
        run_dir.glob("checkpoint_*"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not checkpoints:
        print(f"No checkpoints found in {run_dir}")
        return

    scores = [
        _score_and_normalize(segments, load_fn(ckpt / "reward_model.pt", obs_space, act_space),
                             normalize, robust)
        for ckpt in checkpoints
    ]
    xlim, ylim = _global_limits(scores)

    n_rows = -(-len(checkpoints) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for ax, ckpt, (true_ret, pred_ret, term_status, pr, sr, _) in zip(axes.flat, checkpoints, scores):
        ax.set_visible(True)
        _draw_scatter_ax(ax, true_ret, pred_ret, term_status,
                         xlim=xlim, ylim=ylim,
                         title=f"{ckpt.name}\nr={pr:.2f}  ρ={sr:.2f}",
                         xlabel="True return", ylabel="Pred return",
                         s=12, alpha=0.7)

    handles, labels = next(ax for ax in axes.flat if ax.get_visible()).get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
    plt.suptitle(f"Reward model scatter — all checkpoints\n{run_dir.name}", fontsize=10)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.show()


# ── Reward curves ─────────────────────────────────────────────────────────────

def plot_reward_curves(trajs, rm, normalize=False, robust=False):
    """True vs predicted reward step-by-step for each episode.

    Two subplots per episode: main (all steps except terminal) and terminal bar.
    """
    pred_per_traj = []
    for traj in trajs:
        obs  = np.array([t.observation for t in traj], dtype=np.float32)
        acts = np.array([t.action      for t in traj], dtype=np.float32)
        ns   = np.array([t.next_status for t in traj], dtype=np.float32)
        dn   = np.array([float(t.done) for t in traj], dtype=np.float32)
        pred_per_traj.append(rm.predict_mean_std(obs, acts, ns, dn))

    all_true = np.concatenate([[t.true_reward for t in traj] for traj in trajs])
    all_pred = np.concatenate([pm for pm, _ in pred_per_traj])
    norm_params = _normalization_params(all_true, all_pred, robust) if normalize else None

    n = len(trajs)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3 * n),
                             gridspec_kw={"width_ratios": [4, 1]})
    if n == 1:
        axes = [axes]

    for ep_idx, (traj, (ax_main, ax_term), (pred_mean, pred_std)) in enumerate(
        zip(trajs, axes, pred_per_traj)
    ):
        true_r = np.array([t.true_reward for t in traj])
        pred_r = pred_mean.copy()
        pred_s = pred_std.copy()

        if norm_params is not None:
            pc, ps, tc, ts = norm_params
            pred_r = (pred_r - pc) / ps * ts + tc
            pred_s = pred_s / ps * ts

        norm_tag = f" (pred {'robust' if robust else 'z-score'} aligned)" if normalize else ""

        t_ax = np.arange(len(true_r) - 1)
        ax_main.plot(t_ax, true_r[:-1], label="True",      color="royalblue", lw=1.5,
                     marker="o", markersize=3, markeredgewidth=0)
        ax_main.plot(t_ax, pred_r[:-1], label="Predicted", color="tomato",    lw=1.5, ls="--",
                     marker="o", markersize=3, markeredgewidth=0)

        true_ret_val = float(sum(t.true_reward for t in traj))
        pred_ret_val = float(pred_mean.sum())
        ax_main.set_title(f"Ep {ep_idx+1}  |  true_return={true_ret_val:.1f}  pred_return={pred_ret_val:.1f}")
        ax_main.set_xlabel("Step")
        ax_main.set_ylabel(f"Reward{norm_tag}")
        ax_main.legend(loc="upper right", fontsize=8)

        ax_term.bar([0, 1], [true_r[-1], pred_r[-1]],
                    color=["royalblue", "tomato"], width=0.5, alpha=0.8)
        ax_term.errorbar([1], [pred_r[-1]], yerr=[pred_s[-1]],
                         fmt="none", color="tomato", capsize=4)
        ax_term.set_xticks([0, 1])
        ax_term.set_xticklabels(["True", "Pred"], fontsize=8)
        ax_term.set_title("Terminal")
        ax_term.axhline(0, color="gray", lw=0.5, ls="--")
        ax_term.set_ylim(-60, 1)

    norm_label = f"{'robust' if robust else 'z-score'} normalized" if normalize else "raw"
    plt.suptitle(f"Reward curves — {norm_label}", y=1.002, fontsize=13)
    plt.tight_layout()
    plt.show()


print("Plot functions loaded OK")
