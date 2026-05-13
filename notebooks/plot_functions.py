import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau


# ── Normalization ─────────────────────────────────────────────────────────────

def _zscore(arr):
    return (arr - arr.mean()) / (arr.std() + 1e-8)

def _robust(arr):
    med = np.median(arr)
    iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
    return (arr - med) / (iqr + 1e-8)

def _normalize(arr, robust=False):
    return _robust(arr) if robust else _zscore(arr)


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
            segments.extend(reversed(traj_segs))
    return segments


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score_segments(segments, rm):
    """Return (true_returns, pred_returns) as np.ndarray."""
    true_returns, pred_returns = [], []
    for seg in segments:
        obs  = np.array([t.observation for t in seg], dtype=np.float32)
        acts = np.array([t.action      for t in seg], dtype=np.float32)
        ns   = np.array([t.next_status for t in seg], dtype=np.float32)
        dn   = np.array([float(t.done) for t in seg], dtype=np.float32)
        pred_mean, _ = rm.predict_mean_std(obs, acts, ns, dn)
        true_returns.append(sum(t.true_reward for t in seg))
        pred_returns.append(float(pred_mean.sum()))
    return np.array(true_returns), np.array(pred_returns)


# ── Scatter plot ──────────────────────────────────────────────────────────────

def plot_scatter(trajs, rm, segment_length=None, normalize=False, robust=False):
    """
    Scatter: true return vs predicted return per segment.

    segment_length : int | None  — None = full episode
    normalize      : bool        — rescale predicted to match true's mean/std (true is untouched)
    robust         : bool        — if normalize, use median/IQR instead of mean/std
    """
    segments = _extract_segments(trajs, segment_length)
    if not segments:
        print(f"No segments of length {segment_length} found.")
        return

    true_ret, pred_ret = _score_segments(segments, rm)

    if normalize:
        if robust:
            pred_center = np.median(pred_ret)
            pred_scale  = np.percentile(pred_ret, 75) - np.percentile(pred_ret, 25) + 1e-8
            true_center = np.median(true_ret)
            true_scale  = np.percentile(true_ret, 75) - np.percentile(true_ret, 25) + 1e-8
        else:
            pred_center, pred_scale = pred_ret.mean(), pred_ret.std() + 1e-8
            true_center, true_scale = true_ret.mean(), true_ret.std() + 1e-8
        pred_ret = (pred_ret - pred_center) / pred_scale * true_scale + true_center

    pearson_r,    _ = pearsonr(true_ret, pred_ret)
    spearman_rho, _ = spearmanr(true_ret, pred_ret)
    kendall_tau,  _ = kendalltau(true_ret, pred_ret)

    seg_label = f"seg={segment_length}" if segment_length is not None else "full episode"
    align_tag = f" ({'robust' if robust else 'z-score'} aligned)" if normalize else ""

    combined = np.concatenate([true_ret, pred_ret])
    span = combined.max() - combined.min() + 1e-8
    lo, hi = combined.min() - 0.05 * span, combined.max() + 0.05 * span

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(true_ret, pred_ret, alpha=0.55, s=25, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y=x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True return")
    ax.set_ylabel(f"Predicted return{align_tag}")
    ax.set_title(
        f"Scatter — {seg_label}  (n={len(segments)})\n"
        f"Pearson r={pearson_r:.3f}   Spearman ρ={spearman_rho:.3f}   Kendall τ={kendall_tau:.3f}"
    )
    ax.legend()
    plt.tight_layout()
    plt.show()
    print(
        f"[{seg_label}] Pearson={pearson_r:.3f}  "
        f"Spearman={spearman_rho:.3f}  Kendall={kendall_tau:.3f}  n={len(segments)}"
    )


# ── Reward curves ─────────────────────────────────────────────────────────────

def plot_reward_curves(trajs, rm, normalize=False, robust=False):
    """
    True vs predicted reward step-by-step for each episode.
    Two subplots per episode (horizontal):
      left  — all steps except the terminal one
      right — terminal step only, with its own y-scale

    normalize : bool — normalize using global stats across all steps/episodes
    robust    : bool — if normalize, use median/IQR instead of z-score
    """
    pred_per_traj = []
    for traj in trajs:
        obs  = np.array([t.observation for t in traj], dtype=np.float32)
        acts = np.array([t.action      for t in traj], dtype=np.float32)
        ns   = np.array([t.next_status for t in traj], dtype=np.float32)
        dn   = np.array([float(t.done) for t in traj], dtype=np.float32)
        pm, ps = rm.predict_mean_std(obs, acts, ns, dn)
        pred_per_traj.append((pm, ps))

    all_true = np.concatenate([[t.true_reward for t in traj] for traj in trajs])
    all_pred = np.concatenate([pm for pm, _ in pred_per_traj])

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

        if normalize:
            if robust:
                true_center = np.median(all_true)
                true_scale  = np.percentile(all_true, 75) - np.percentile(all_true, 25) + 1e-8
                pred_center = np.median(all_pred)
                pred_scale  = np.percentile(all_pred, 75) - np.percentile(all_pred, 25) + 1e-8
            else:
                true_center, true_scale = all_true.mean(), all_true.std() + 1e-8
                pred_center, pred_scale = all_pred.mean(), all_pred.std() + 1e-8
            # true_r stays in its original scale; only predicted is rescaled to match it
            pred_r = (pred_r - pred_center) / pred_scale * true_scale + true_center
            pred_s = pred_s / pred_scale * true_scale

        norm_tag = f" (pred {'robust' if robust else 'z-score'} aligned)" if normalize else ""

        # ── left: all steps except terminal ──────────────────────────────
        t_ax = np.arange(len(true_r) - 1)
        ax_main.plot(t_ax, true_r[:-1], label="True",     color="royalblue", lw=1.5,
                     marker="o", markersize=3, markeredgewidth=0)
        ax_main.plot(t_ax, pred_r[:-1], label="Predicted", color="tomato",    lw=1.5, ls="--",
                     marker="o", markersize=3, markeredgewidth=0)
        ax_main.fill_between(t_ax,
                             pred_r[:-1] - pred_s[:-1],
                             pred_r[:-1] + pred_s[:-1],
                             alpha=0.2, color="tomato", label="±1σ ensemble")

        true_ret_val = float(sum(t.true_reward for t in traj))
        pred_ret_val = float(pred_mean.sum())
        ax_main.set_title(f"Ep {ep_idx+1}  |  true_return={true_ret_val:.1f}  pred_return={pred_ret_val:.1f}")
        ax_main.set_xlabel("Step")
        ax_main.set_ylabel(f"Reward{norm_tag}")
        ax_main.legend(loc="upper right", fontsize=8)

        # ── right: terminal step only ─────────────────────────────────────
        ax_term.bar([0, 1], [true_r[-1], pred_r[-1]],
                    color=["royalblue", "tomato"], width=0.5, alpha=0.8)
        ax_term.errorbar([1], [pred_r[-1]], yerr=[pred_s[-1]],
                         fmt="none", color="tomato", capsize=4)
        ax_term.set_xticks([0, 1])
        ax_term.set_xticklabels(["True", "Pred"], fontsize=8)
        ax_term.set_title("Terminal")
        ax_term.axhline(0, color="gray", lw=0.5, ls="--")

    norm_label = f"{'robust' if robust else 'z-score'} normalized" if normalize else "raw"
    plt.suptitle(f"Reward curves — {norm_label}", y=1.002, fontsize=13)
    plt.tight_layout()
    plt.show()


print("Plot functions loaded OK")
