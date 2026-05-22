import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau, weightedtau

STATUS_RUNNING = 4

STATUS_LABELS = {
    0: "arrived", 1: "collision", 2: "off_road", 3: "timeout",
    4: "running", 5: "teleported", 6: "removed",
}
STATUS_COLORS = {
    0: "steelblue", 1: "orange", 2: "crimson", 3: "mediumpurple",
    4: "gray",      5: "gold",   6: "saddlebrown",
}


# ── helper metrics ─────────────────────────────────────────────────────────────

def ccc(x, y):
    """Lin's Concordance Correlation Coefficient."""
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x,  var_y  = np.var(x),  np.var(y)
    cov = np.cov(x, y, ddof=0)[0, 1]
    return (2 * cov) / (var_x + var_y + (mean_x - mean_y) ** 2)


def plot_test(segments, reward_model, norm_on_running, matching_mean, plot_on_running, matching_std):
    all_transitions = [t for segment in segments for t in segment]

    true_rewards = np.array([t.true_reward  for t in all_transitions], dtype=np.float32)
    obs    = np.array([t.observation for t in all_transitions], dtype=np.float32)
    acts   = np.array([t.action      for t in all_transitions], dtype=np.float32)
    status = np.array([t.next_status for t in all_transitions], dtype=np.float32)
    done   = np.array([float(t.done) for t in all_transitions], dtype=np.float32)
    pred_rewards = reward_model.predict(obs, acts, status, done)

    # ── normalisation mask ────────────────────────────────────────────────────
    norm_mask = np.ones(len(all_transitions), dtype=bool)
    if norm_on_running:
        norm_mask = status[:, STATUS_RUNNING] == 1

    true_mean = np.mean(true_rewards[norm_mask])
    true_std  = np.std(true_rewards[norm_mask])
    pred_mean = np.mean(pred_rewards[norm_mask])
    pred_std  = np.std(pred_rewards[norm_mask])

    if matching_mean and matching_std:
        pred_rewards_norm = (pred_rewards - pred_mean) / pred_std * true_std + true_mean
    elif matching_mean:
        pred_rewards_norm = pred_rewards - pred_mean + true_mean
    else:
        pred_rewards_norm = pred_rewards

    # ── correlations ──────────────────────────────────────────────────────────
    pr,  _ = pearsonr(true_rewards, pred_rewards_norm)
    sr,  _ = spearmanr(true_rewards, pred_rewards_norm)
    kt,  _ = kendalltau(true_rewards, pred_rewards_norm)

    # ── new metrics ───────────────────────────────────────────────────────────
    ccc_val  = ccc(true_rewards, pred_rewards_norm)
    rmse_val = np.sqrt(np.mean((pred_rewards_norm - true_rewards) ** 2))
    mae_val  = np.mean(np.abs(pred_rewards_norm - true_rewards))
    wt_val,  _ = weightedtau(true_rewards, pred_rewards_norm)   # weighted Kendall

    # ── plot mask ─────────────────────────────────────────────────────────────
    plot_mask = np.ones(len(pred_rewards_norm), dtype=bool)
    if plot_on_running:
        plot_mask = status[:, STATUS_RUNNING] == 1

    pred_rewards_plot = pred_rewards_norm[plot_mask]
    true_rewards_plot = true_rewards[plot_mask]
    terminal_status   = np.argmax(status, axis=1)

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))

    for status_id, color in STATUS_COLORS.items():
        m = terminal_status[plot_mask] == status_id
        if not m.any():
            continue
        ax.scatter(
            true_rewards_plot[m], pred_rewards_plot[m],
            color=color, label=STATUS_LABELS[status_id],
            alpha=0.7, s=20, edgecolors="none",
        )

    lo = min(true_rewards_plot.min(), pred_rewards_plot.min())
    hi = max(true_rewards_plot.max(), pred_rewards_plot.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x")

    ref_tag  = "running ref" if norm_on_running else "all ref"
    norm_tag = (f"z-score ({ref_tag})"    if (matching_mean and matching_std) else
                f"mean-shift ({ref_tag})" if matching_mean else "raw")

    ax.set_xlabel("True reward")
    ax.set_ylabel(f"Pred reward ({norm_tag})")

    # ── two-line title: classic metrics + new metrics ─────────────────────────
    line1 = f"Pearson r={pr:.3f}   Spearman ρ={sr:.3f}   Kendall τ={kt:.3f}   n={len(all_transitions)}"
    line2 = f"CCC={ccc_val:.3f}   RMSE={rmse_val:.4f}   MAE={mae_val:.4f}   Weighted-τ={wt_val:.3f}"
    ax.set_title(f"{line1}\n{line2}", fontsize=8.5)

    ax.legend(fontsize=8, markerscale=1.2)
    plt.tight_layout()
    plt.show()

    return fig, ax