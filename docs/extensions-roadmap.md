# Roadmap estensioni e sequenza sperimentale

Documento di riferimento per le fasi successive alla campagna di tuning.
I punti del codice citati contengono placeholder commentati marcati
`EXTENSION PLACEHOLDER`: descrivono l'integrazione prevista senza cambiare in
alcun modo il comportamento attuale.

## Sequenza sperimentale

1. Completare il tuning in corso (6 bracci, project `tuning-thesis`).
2. Selezionare e **salvare** i migliori iperparametri:
   `python scripts/export_best_config.py --arm <arm> --save-dir configs/best`
   (un JSON per braccio con parametri, override completi, objective e
   provenienza — indipendente dal journal Optuna; committare `configs/best/`).
3. Curve di budget (`run_budget_curves.sh`, report con
   `report_budget_curves.py` → budget minimi X e Y).
4. Run robuste a 5 seed (`run_final_5seeds.sh`, project `thesis`;
   strategie A = X/2+Y/2 e B = X+Y per gli hybrid).
5. Implementare ADAM per il `demo_weight` (placeholder sotto).
6. Verificare ADAM su hybrid + preferenze soft; se positivo, estendere.
7. Tuning di **hybrid + preferenze Bernoulli** (nuovo braccio; vedi la nota
   `initial_queries` sotto).
8. Confronto con le baseline singole e con la baseline ibrida di letteratura
   (Ibarz 2018, sotto).
9. Esperimenti con rumore su preferenze e dimostrazioni (placeholder sotto).

Criterio di chiusura della tesi: risultati robusti (5 seed + curve di budget)
in cui hybrid batte le baseline singole sia con preferenze soft sia Bernoulli.

## 1. ADAM per il demo_weight

Placeholder in `human_feedback_rl/algorithms/hybrid_algorithm.py`
(`__init__` e `_reward_step`). Design previsto:

- `demo_weight` diventa un parametro appreso in log-spazio
  (`log_demo_weight = th.nn.Parameter(log(w0))`, quindi `w = exp(·) > 0`),
  con un Adam dedicato (`demo_weight_lr`, nuovo kwarg).
- In `_reward_step` la `scale` usa `exp(log_demo_weight)`; dopo l'update dei
  pesi della reward net, il peso fa il suo passo Adam su un segnale di
  validazione: la loss Bradley-Terry del membro aggiornato su un batch
  held-out (`dataset_val`), con gradiente rispetto a `log_demo_weight`
  ottenuto differenziando attraverso il coefficiente di mixing (one-step
  unrolled oppure differenze finite).
- Feature disattivata di default: con il flag off il comportamento deve
  restare bit-identico al peso costante (il braccio tunato resta valido).
- Primo test: hybrid + preferenze soft, confronto contro il `demo_weight`
  costante trovato dal tuning a parità di tutto il resto.

## 2. Rumore nei dati

- **Preferenze** — `human_feedback_rl/common/gatherers.py`
  (`PreferenceGathererFromReward`): nuovo knob `label_noise` che con
  probabilità data scambia/flippa l'etichetta prodotta dall'oracolo.
  Separato da `temperature` (softness ≠ errori). `label_noise=0` = identico.
- **Dimostrazioni** — `scripts/_common.py` (`load_expert_trajectories`):
  nuovo knob `demo_noise` applicato dopo il sottocampionamento — rumore
  gaussiano clippato sulle azioni dell'esperto oppure sostituzione di una
  frazione di traiettorie con traiettorie di qualità agente. Il caricamento è
  l'unico punto di passaggio, quindi la corruzione vale sia per le loss IRL
  sia per le coppie demos-as-preferences. `demo_noise=0` = identico.

## 3. Baseline ibrida di letteratura (Ibarz et al. 2018)

"Dimostrazioni come preferenze implicite" è **già implementata e testata**:
`HybridAlgorithm(demo_mode="preferences")` — le demo entrano come coppie
(frammento esperto ≻ frammento agente) mescolate alle query dell'oracolo in
un unico obiettivo Bradley-Terry (`dataset_demo_prefs_*`,
`_collect_demo_preference_pairs`, `_train_reward_model_pure_preferences`).
Per usarla come braccio di confronto: placeholder `ibarz` in
`scripts/tune_hybrid_sac.py` (riusa i parametri pref + tuning di
`demo_pref_batch_fraction`; tutto il downstream — pruning, export, run
finali, report — funziona senza modifiche).

## 4. Nota per la fase 7: `initial_queries` di hybrid Bernoulli

Oggi i bracci hybrid **fissano** `initial_queries = 10% del budget pref`
(500 a 5000), mentre i bracci pref-only lo **tunano** (scelte 2–20% del
budget). Non ci sono dipendenze implicite dal valore assoluto 500: è già
derivato dal budget, viene scorporato dal totale (`build_query_schedule`),
e il vincolo `initial_queries ≤ total_queries` è garantito; il confronto
resta equo perché il budget totale X non cambia.

Decisione da prendere prima del tuning di hybrid_bernoulli (raccomandazione):
**tunare `initial_queries`** con le stesse frazioni 2–20% del budget usate per
pref_bernoulli — con etichette rumorose la dimensione del bootstrap è
plausibilmente più critica che con le soft, e il 10% fisso sarebbe
un'assunzione non verificata. Ricordare inoltre il budget: per Bernoulli il
budget di riferimento è 100K query (`--pref-budget 100000`,
`--study-suffix` dedicato), non i 5000 dei bracci soft.
