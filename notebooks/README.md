# Notebooks

Analisi e provenienza dei dataset per la tesi. Tutti i notebook usano percorsi
**relativi alla repo** (vanno eseguiti dalla cartella `notebooks/`) e leggono i
pickle da `datasets/` (scaricabili con `scripts/download_datasets.py`).

| Notebook | Scopo | Richiede SUMO |
|---|---|---|
| `reward_model_scatter_analysis.ipynb` | **Analisi qualità del reward model** di un checkpoint: scatter true-vs-predicted (livello transizione e traiettoria, per esito terminale), correlazioni (Pearson/Spearman/Kendall/CCC), curve reward-per-step per episodio. È il notebook da usare per le figure di analisi della tesi. | Sì (per il rollout dell'agente; le sezioni su dataset fissi girano anche senza) |
| `create_dataset_for_correlation.ipynb` | **Provenienza** di `datasets/debug_dataset_full_ep.pkl`: campiona un rollout da un checkpoint e ne estrae 80 episodi bilanciati per esito (20 per classe). Rieseguirlo produce un file nuovo, mai sovrascrive. | Sì |
| `extract_debug_episodes.ipynb` | **Provenienza** della conversione `debug_dataset.pkl` (transizioni piatte) → `debug_dataset_full_ep.pkl` (episodi completi), con sanity check di conservazione. | No |
| `remove_collision_expert_trajectories.ipynb` | **Provenienza** di `datasets/expert_trajectories_no_collision.pkl`: filtra le traiettorie esperte che terminano in collisione dal pickle grezzo. | No |

Helper condiviso:

- `loadings.py` — `load_reward_ensemble()`: ricostruisce una `RewardEnsemble`
  da un `reward_model.pt` ispezionando lo state dict (architettura inferita;
  supporta anche i layout dei checkpoint più vecchi).

Nota: le diagnostiche *durante* il training (correlazioni pred↔true, scatter
periodici, accuratezza sulle preferenze) sono già loggate su W&B da
`HybridAlgorithm`; questi notebook servono per l'analisi *post-hoc* di un
checkpoint specifico e per documentare come sono nati i dataset fissi.
