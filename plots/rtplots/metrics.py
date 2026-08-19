"""Catalogo delle metriche plottabili.

Due generi, non due fonti come nel progetto di ispirazione (che distingueva
`.npz` locali da history W&B: qui non ci sono file locali raggiungibili, e' tutto
W&B, vedi `paths.py`):

  - `curve`   una serie storica per run (valore per iterazione/timestep),
              aggregata sui seed in `curves.py` -> curve di apprendimento;
  - `summary` un solo valore finale per run (`run.summary[...]`, scritto da
              `log_sweep_summary` a fine training), aggregato sui seed per
              livello di budget in `budget.py` -> curve di budget.

Le curve hanno **due assi x diversi** a seconda di chi le logga (verificato sui
dati reali di `tuning-thesis-budget-curves-completion`, non assunto):
  - l'agente SAC logga sotto il prefisso `agent/` (via `PrefixedLogger`), il suo
    asse naturale e' `agent/time/total_timesteps`;
  - l'algoritmo di reward learning logga tutto il resto (`reward/*`,
    `rollout/*`, `reward_val/*`, `imitation/*`) senza prefisso, sul contatore
    `iterations` (vedi `base_reward_learning_algorithm.py:record("iterations", ...)`).
Usare l'asse sbagliato per una metrica la farebbe sembrare vuota (tutti NaN),
quindi ogni voce del catalogo porta il proprio `step_key`.
"""
from __future__ import annotations

AGENT_STEP = "agent/time/total_timesteps"
ITER_STEP = "iterations"

# (gruppo, [(chiave, etichetta UI, etichetta asse y, genere, step_key)])
METRIC_GROUPS = [
    ("Eval finale (budget curves)", [
        ("sweep/mean_fast_return", "Return (fast)", "mean_fast_return", "summary", None),
        ("sweep/mean_comfort_return", "Return (comfort)", "mean_comfort_return", "summary", None),
        ("sweep/mean_speed", "Velocità media", "mean_speed", "summary", None),
        ("sweep/mean_ep_length", "Lunghezza episodio", "mean_ep_length", "summary", None),
        ("sweep/success_rate", "Success rate", "success_rate", "summary", None),
        ("sweep/collision_rate", "Collision rate", "collision_rate", "summary", None),
        ("sweep/off_road_rate", "Off-road rate", "off_road_rate", "summary", None),
        ("sweep/timeout_rate", "Timeout rate", "timeout_rate", "summary", None),
    ]),
    ("Agente (curva di apprendimento)", [
        ("agent/rewards/ep_fast_return", "Return (fast, training)", "ep_fast_return",
         "curve", AGENT_STEP),
        ("agent/rewards/ep_comfort_return", "Return (comfort, training)", "ep_comfort_return",
         "curve", AGENT_STEP),
        ("agent/rewards/ep_env_return", "Return ambiente", "ep_env_return", "curve", AGENT_STEP),
        ("agent/rollout/ep_rew_mean", "Return di rollout (SB3)", "ep_rew_mean",
         "curve", AGENT_STEP),
        ("agent/performance/ep_avg_speed", "Velocità media episodio", "ep_avg_speed",
         "curve", AGENT_STEP),
        ("agent/event_rate/successes", "Success rate (training)", "successes",
         "curve", AGENT_STEP),
        ("agent/event_rate/collisions", "Collision rate (training)", "collisions",
         "curve", AGENT_STEP),
        ("agent/event_rate/off_road", "Off-road rate (training)", "off_road",
         "curve", AGENT_STEP),
        ("agent/event_rate/timeouts", "Timeout rate (training)", "timeouts",
         "curve", AGENT_STEP),
        ("agent/train/actor_loss", "Actor loss", "actor_loss", "curve", AGENT_STEP),
        ("agent/train/critic_loss", "Critic loss", "critic_loss", "curve", AGENT_STEP),
        ("agent/train/ent_coef", "Entropy coefficient", "ent_coef", "curve", AGENT_STEP),
    ]),
    ("Reward model (diagnostica)", [
        ("reward/loss", "Loss totale", "loss", "curve", ITER_STEP),
        ("reward/loss_pref_train", "Loss BT (train)", "loss_pref_train", "curve", ITER_STEP),
        ("reward/loss_pref_val", "Loss BT (val)", "loss_pref_val", "curve", ITER_STEP),
        ("reward/acc_pref_train", "Accuratezza BT (train)", "acc_pref_train", "curve", ITER_STEP),
        ("reward/acc_pref_val", "Accuratezza BT (val)", "acc_pref_val", "curve", ITER_STEP),
        ("reward/grad_norm", "Norma gradiente", "grad_norm", "curve", ITER_STEP),
        ("reward/weight_norm", "Norma pesi", "weight_norm", "curve", ITER_STEP),
        ("reward/expert_model_margin", "Margine esperto/modello", "expert_model_margin",
         "curve", ITER_STEP),
        ("reward/expert_return_mean", "Return predetto (esperto)", "expert_return_mean",
         "curve", ITER_STEP),
        ("reward/model_return_mean", "Return predetto (agente)", "model_return_mean",
         "curve", ITER_STEP),
        # solo hybrid: due loss separate + i due gradienti che demo_weight bilancia
        ("reward/hybrid_demo_loss", "Loss demo (hybrid)", "hybrid_demo_loss",
         "curve", ITER_STEP),
        ("reward/hybrid_pref_loss", "Loss preferenze (hybrid)", "hybrid_pref_loss",
         "curve", ITER_STEP),
        ("reward/grad_norm_demo_pref_ratio", "Rapporto gradienti demo/pref",
         "grad_norm_demo_pref_ratio", "curve", ITER_STEP),
    ]),
    ("Validazione reward (correlazione pred/true)", [
        ("reward_val/current_rollout/post_update/pred_true/pearson_all",
         "Pearson pred/true (rollout corrente)", "pearson", "curve", ITER_STEP),
        ("reward_val/current_rollout/post_update/pred_true/spearman_all",
         "Spearman pred/true (rollout corrente)", "spearman", "curve", ITER_STEP),
        ("reward_val/debug_dataset/post_update/pred_true/pearson_all",
         "Pearson pred/true (dataset di debug)", "pearson", "curve", ITER_STEP),
        ("reward_val/debug_dataset/post_update/pred_true/spearman_all",
         "Spearman pred/true (dataset di debug)", "spearman", "curve", ITER_STEP),
        ("reward_val/current_rollout/post_update/gap_arrived_collided",
         "Divario reward arrivato/collisione", "gap_arrived_collided", "curve", ITER_STEP),
    ]),
    # Stimatore di alpha introdotto dopo il meeting del 2026-08-06: la varianza
    # e' quella del gradiente indotto dal SINGOLO campione attorno al gradiente
    # medio, /(N-1) e poi /B. Prima si misurava la dispersione FRA probe batch,
    # che dipende da quanto sono grandi i batch e non risponde alla domanda
    # "quanto cambierebbe il gradiente con altri campioni". Le chiavi `alpha/`
    # esistono solo nelle run lanciate dopo quel cambiamento: sulle precedenti
    # la curva risulta vuota, che e' corretto.
    ("Stima di α — varianza di campionamento", [
        ("reward/hybrid_alpha", "Peso delle dimostrazioni (α)", "alpha",
         "curve", ITER_STEP),
        ("reward/hybrid_alpha_active", "α stimato (1) o fissato a 1 (0)", "attivo",
         "curve", ITER_STEP),
        ("alpha/S_pref", "Var. della media — preferenze (S_p)", "S_pref",
         "curve", ITER_STEP),
        ("alpha/S_demo", "Var. della media — dimostrazioni (S_d)", "S_demo",
         "curve", ITER_STEP),
        ("alpha/cv2_pref", "CV² — preferenze", "cv2_pref", "curve", ITER_STEP),
        ("alpha/cv2_demo", "CV² — dimostrazioni", "cv2_demo", "curve", ITER_STEP),
        ("alpha/V_pref", "Var. del processo — preferenze (V_p)", "V_pref",
         "curve", ITER_STEP),
        ("alpha/V_demo", "Var. del processo — dimostrazioni (V_d)", "V_demo",
         "curve", ITER_STEP),
        ("alpha/gradmean_norm_sq_pref", "‖gradiente medio‖² — preferenze",
         "gradmean_norm_sq_pref", "curve", ITER_STEP),
        ("alpha/gradmean_norm_sq_demo", "‖gradiente medio‖² — dimostrazioni",
         "gradmean_norm_sq_demo", "curve", ITER_STEP),
        ("alpha/n_pref", "Campioni usati (N_p)", "n_pref", "curve", ITER_STEP),
        ("alpha/n_demo", "Campioni usati (N_d)", "n_demo", "curve", ITER_STEP),
        ("alpha/batch_pref", "Minibatch (B_p)", "batch_pref", "curve", ITER_STEP),
        ("alpha/batch_demo", "Minibatch (B_d)", "batch_demo", "curve", ITER_STEP),
    ]),
    # Loggate solo dalle run di thesis-grad-diagnostics che girano con
    # `grad_probe_batches > 0`: K minibatch diagnostici indipendenti valutati
    # agli stessi parametri a fine iterazione. Nelle altre campagne queste
    # chiavi non esistono e la curva risulta vuota, che e' corretto.
    #
    # Da agosto 2026 questi probe NON alimentano piu' alpha: restano
    # diagnostica, la stima vive nel gruppo qui sopra.
    ("Gradienti — frozen probe (grad-diagnostics)", [
        ("reward/grad_probe_dir_var_pref", "Var. direzionale — preferenze (CV_p²)",
         "dir_var_pref", "curve", ITER_STEP),
        ("reward/grad_probe_dir_var_demo", "Var. direzionale — dimostrazioni (CV_d²)",
         "dir_var_demo", "curve", ITER_STEP),
        ("reward/grad_probe_precond_dir_var_pref",
         "Var. direzionale post-Adam — preferenze", "dir_var_pref", "curve", ITER_STEP),
        ("reward/grad_probe_precond_dir_var_demo",
         "Var. direzionale post-Adam — dimostrazioni", "dir_var_demo", "curve", ITER_STEP),
        ("reward/grad_probe_cosine_of_means", "Coseno fra i gradienti medi",
         "coseno", "curve", ITER_STEP),
        ("reward/grad_probe_cosine", "Coseno per campione", "coseno", "curve", ITER_STEP),
        ("reward/grad_probe_var_pref", "Varianza — preferenze", "var_pref",
         "curve", ITER_STEP),
        ("reward/grad_probe_var_demo", "Varianza — dimostrazioni", "var_demo",
         "curve", ITER_STEP),
        ("reward/grad_probe_mean_sq_norm_pref", "Norma quadratica — preferenze",
         "mean_sq_norm_pref", "curve", ITER_STEP),
        ("reward/grad_probe_mean_sq_norm_demo", "Norma quadratica — dimostrazioni",
         "mean_sq_norm_demo", "curve", ITER_STEP),
        ("reward/demo_2_expert_softmax_mass", "Massa softmax sugli esperti (demo_2)",
         "expert_softmax_mass", "curve", ITER_STEP),
    ]),
    # Il reward consegnato all'agente, non quello con cui si allena il modello:
    # e' qui che si vede la normalizzazione (sigma = 1 esatto quando e' attiva).
    ("Normalizzazione del reward", [
        ("reward_val/current_rollout/post_update/reward_std",
         "Sigma del reward visto dall'agente", "reward_std", "curve", ITER_STEP),
        ("reward_val/current_rollout/post_update/reward_mean",
         "Media del reward visto dall'agente", "reward_mean", "curve", ITER_STEP),
        ("reward/normalization_raw_std", "Sigma grezzo del modello (solo norm ON)",
         "raw_std", "curve", ITER_STEP),
        ("replay_relabel_debug/delta_abs_mean",
         "Scarto buffer vs ricalcolato", "delta_abs_mean", "curve", ITER_STEP),
        ("replay_relabel_debug/current_reward_std",
         "Sigma del reward nel replay buffer", "current_reward_std", "curve", ITER_STEP),
    ]),
    ("Rollout (raccolta feedback)", [
        ("rollout/mean_true_reward", "Reward vero medio", "mean_true_reward",
         "curve", ITER_STEP),
        ("rollout/mean_model_reward", "Reward predetto medio", "mean_model_reward",
         "curve", ITER_STEP),
        ("rollout/mean_length", "Lunghezza traiettoria", "mean_length", "curve", ITER_STEP),
    ]),
    ("Dimostrazioni (imitation)", [
        ("imitation/action_rmse", "RMSE azione vs esperto", "action_rmse",
         "curve", ITER_STEP),
        ("imitation/expert_action_nll", "NLL azione esperta", "expert_action_nll",
         "curve", ITER_STEP),
    ]),
]

METRICS = {
    key: {"key": key, "label": label, "ylabel": ylabel, "kind": kind, "step_key": step_key}
    for _, items in METRIC_GROUPS for key, label, ylabel, kind, step_key in items
}

DEFAULT_CURVE_METRIC = "agent/rewards/ep_fast_return"
DEFAULT_SUMMARY_METRIC = "sweep/mean_fast_return"


def metric_info(key: str) -> dict:
    """Voce del catalogo; per una chiave W&B non elencata restituisce un default
    sensato, indovinando genere e asse x dal prefisso (`sweep/` = summary,
    `agent/` = curva sui timestep, il resto = curva sulle iterazioni)."""
    if key in METRICS:
        return METRICS[key]
    if key.startswith("sweep/"):
        kind, step_key = "summary", None
    elif key.startswith("agent/"):
        kind, step_key = "curve", AGENT_STEP
    else:
        kind, step_key = "curve", ITER_STEP
    return {"key": key, "label": key, "ylabel": key.split("/")[-1].replace("_", " "),
            "kind": kind, "step_key": step_key}


def ui_groups(kind: str | None = None) -> list[dict]:
    """Struttura per le tendine della pagina, filtrabile per genere (curve/summary)."""
    out = []
    for group, items in METRIC_GROUPS:
        options = [{"key": k, "label": lab, "kind": knd}
                   for k, lab, _, knd, _ in items if kind is None or knd == kind]
        if options:
            out.append({"group": group, "options": options})
    return out
