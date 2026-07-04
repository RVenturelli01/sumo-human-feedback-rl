# Demonstration-Based Reward Learning — MaxEnt IRL

**Pseudocodice Matematico dell'Algoritmo `DemoAlgorithm`**

*Analisi del file `demo_algorithm.py` e della base class `base_reward_learning_algorithm.py`*

---

## 1. Panoramica

Il **`DemoAlgorithm`** è un metodo di *reward learning from demonstrations* basato sul principio del **Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL)**. A differenza degli approcci basati su preferenze (es. Christiano et al.), non richiede label scalari né confronti tra frammenti: l'esperto fornisce un insieme fisso di traiettorie complete passate all'inizializzazione e non viene più consultato.

L'idea centrale è:

- L'esperto ha già generato un set $\mathcal{T}^E$ di traiettorie; queste rimangono fisse per tutta la sessione.
- Ad ogni iterazione, l'agente genera nuove traiettorie $\mathcal{T}^M$ (le "model trajectories").
- Il reward model è aggiornato minimizzando la **perdita MaxEnt IRL**:

$$
\mathcal{L}(\boldsymbol{\theta}) = -\underbrace{\operatorname{mean}_{\tau^E}\bigl[R_{\boldsymbol{\theta}}(\tau^E)\bigr]}_{\text{massimizza return esperto}} + \underbrace{\operatorname{logsumexp}_{\tau^M}\bigl[R_{\boldsymbol{\theta}}(\tau^M)\bigr] - \log M_m}_{\approx\,\log Z_{\boldsymbol{\theta}} \;\text{(funzione di partizione)}}
$$

- Il reward model appreso viene normalizzato (solo la media) e usato da PPO per addestrare la policy.

> **Schema del ciclo esterno:**
> *(Pre-warm opzionale)* → *Rollout agente* → *Aggiorna RM (MaxEnt IRL)* → *Normalizza RM (media)* → *Addestra policy (PPO)*

---

## 2. Strutture Dati Fondamentali

### 2.1 Transizione

Una **transizione** è il dato atomico del sistema:

$$
t = \bigl(o_t,\; a_t,\; r_t^{\mathrm{true}},\; \boldsymbol{s}_{t+1},\; \mathrm{done}_t\bigr)
$$

dove:

- $o_t \in \mathbb{R}^{d_\mathrm{obs}}$: osservazione all'istante $t$.
- $a_t \in \mathbb{R}^{d_\mathrm{act}}$: azione eseguita.
- $r_t^{\mathrm{true}} \in \mathbb{R}$: reward vero dell'ambiente (non usato per il training del RM, solo per logging e correlazione).
- $\boldsymbol{s}_{t+1} \in \{0,1\}^{7}$: vettore one-hot dello stato terminale (`arrived, collided, off_road, timeout, running, teleported, removed_unknown`).
- $\mathrm{done}_t \in \{0,1\}$: flag di terminazione episodio.

### 2.2 Traiettoria

Una **traiettoria** $\tau = (t_1, t_2, \ldots, t_T)$ è una sequenza ordinata di transizioni prodotta eseguendo la policy sull'ambiente fino alla terminazione dell'episodio.

Il **return totale** di una traiettoria con il reward model è:

$$
R_{\boldsymbol{\theta}}(\tau) = \sum_{j=1}^{T} \mathrm{member}\bigl(o_j,\, a_j,\, \boldsymbol{s}_j,\, d_j\bigr)
$$

Nota: le traiettorie esperto $\mathcal{T}^E$ sono **fisse** (passate al costruttore), mentre le traiettorie modello $\mathcal{T}^M$ sono **fresche** ad ogni iterazione (rollout dalla policy corrente).

---

## 3. Architettura del Reward Model

### 3.1 SumoRewardNet (rete base)

Rete neurale MLP che mappa ogni transizione in uno scalare:

$$
r_{\boldsymbol{\theta}} : \mathbb{R}^{d_\mathrm{obs}} \times \mathbb{R}^{d_\mathrm{act}} \times \mathbb{R}^7 \times \mathbb{R} \;\longrightarrow\; \mathbb{R}
$$

$$
r_{\boldsymbol{\theta}}(o, a, \boldsymbol{s}, d) = \mathrm{MLP}\bigl([o \,\|\, a \,\|\, \boldsymbol{s} \,\|\, d]\bigr)
$$

Architettura di default: due layer nascosti da 128 unità con attivazione $\tanh$.  
Dimensione input: $d_\mathrm{obs} + d_\mathrm{act} + 7 + 1$.

### 3.2 NormalizedRewardNet (wrapper di normalizzazione)

Avvolge una rete e applica una sottrazione della media centrata. Mantiene scalari $(\mu, \sigma)$ aggiornabili via EMA con fattore $\alpha \in (0,1]$:

$$
\mu \;\leftarrow\; (1-\alpha)\,\mu + \alpha\,\mu_{\mathrm{new}}, \qquad \sigma \;\leftarrow\; (1-\alpha)\,\sigma + \alpha\,\sigma_{\mathrm{new}}
$$

Con $\alpha = 1$ (default) la EMA degenera in un'assegnazione diretta.

Comportamento in forward vs predict:

$$
\text{forward (training/loss):}\quad \tilde{r}(o,a,\boldsymbol{s},d) = r_{\boldsymbol{\theta}}(o,a,\boldsymbol{s},d) - \mu
$$

$$
\text{predict (PPO + validazione):}\quad \tilde{r}(o,a,\boldsymbol{s},d) = \frac{r_{\boldsymbol{\theta}}(o,a,\boldsymbol{s},d) - \mu}{\sigma + \varepsilon}
$$

### 3.3 RewardEnsemble

Un insieme di $K$ reti indipendenti. Ciascun membro è già un `NormalizedRewardNet(SumoRewardNet)`. L'output dell'ensemble è la media:

$$
r_{\boldsymbol{\Theta}}(o,a,\boldsymbol{s},d) = \frac{1}{K}\sum_{k=1}^K \tilde{r}_k(o,a,\boldsymbol{s},d)
$$

### 3.4 Architettura completa (double-wrapped)

```
reward_model = NormalizedRewardNet(                     # outer wrapper (μ_outer, σ_outer)
    RewardEnsemble([
        NormalizedRewardNet(SumoRewardNet(...)),         # member_1 (μ_1, σ_1)
        ...
        NormalizedRewardNet(SumoRewardNet(...)),         # member_K (μ_K, σ_K)
    ])
)
```

La **forward pass del training** opera sui singoli membri (`member_k.forward(...)`):

$$
\mathrm{member}_k\bigl(o,a,\boldsymbol{s},d\bigr) = r_{\boldsymbol{\theta}_k}(o,a,\boldsymbol{s},d) - \mu_k
$$

La **forward pass per PPO** usa l'intero `reward_model`:

$$
\mathrm{reward\_model.predict}(o,a,\boldsymbol{s},d) = \frac{\left[\frac{1}{K}\sum_k (r_{\boldsymbol{\theta}_k}(o,a,\boldsymbol{s},d) - \mu_k)\right] - \mu_{\mathrm{outer}}}{\sigma_{\mathrm{outer}} + \varepsilon}
$$

---

## 4. Formulazione Matematica della Perdita (MaxEnt IRL)

### 4.1 Return totale di una traiettoria

Dato un membro $k$ e una traiettoria $\tau = (t_1, \ldots, t_T)$:

$$
R_k(\tau) = \sum_{j=1}^{T} \mathrm{member}_k\bigl(o_j,\, a_j,\, \boldsymbol{s}_j,\, d_j\bigr) = \sum_{j=1}^{T} \bigl(r_{\boldsymbol{\theta}_k}(o_j, a_j, \boldsymbol{s}_j, d_j) - \mu_k\bigr)
$$

### 4.2 Perdita MaxEnt IRL

Per ciascun membro $k$, estratto un mini-batch di $n_e$ traiettorie esperto e $n_m$ traiettorie modello:

$$
\mathcal{L}_k(\boldsymbol{\theta}_k) = -\frac{1}{n_e}\sum_{i=1}^{n_e} R_k(\tau^E_i) + \underbrace{\log\!\left(\frac{1}{n_m}\sum_{j=1}^{n_m} e^{R_k(\tau^M_j)}\right)}_{\operatorname{logsumexp}(R_k(\mathcal{T}^M)) - \log n_m}
$$

Forma compatta (equivalente numericamente stabile):

$$
\mathcal{L}_k = -\operatorname{mean}\bigl[R_k(\mathcal{T}^E_{\mathrm{batch}})\bigr] + \operatorname{logsumexp}\bigl[R_k(\mathcal{T}^M_{\mathrm{batch}})\bigr] - \log n_m
$$

### 4.3 Interpretazione del gradiente

$$
\nabla_{\boldsymbol{\theta}_k}\mathcal{L}_k = -\underbrace{\frac{1}{n_e}\sum_i \nabla_{\boldsymbol{\theta}_k} R_k(\tau^E_i)}_{\text{features medie dell'esperto}} + \underbrace{\sum_j \underbrace{\frac{e^{R_k(\tau^M_j)}}{\sum_{j'} e^{R_k(\tau^M_{j'})}}}_{\text{softmax weight}} \cdot \nabla_{\boldsymbol{\theta}_k} R_k(\tau^M_j)}_{\text{features medie sotto exp}(R_k)}
$$

Il reward model è spinto ad assegnare return alti all'esperto e a controllare la "funzione di partizione" attraverso le traiettorie modello fresche dalla policy corrente (no importance sampling necessario).

---

## 5. Pseudocodice Completo

### 5.1 Loop Esterno Principale

> **Algorithm 1 — `DemoAlgorithm.train` (Loop principale)**
>
> **Input:** Env $\mathcal{E}$, agente PPO $\pi$, traiettorie esperto fisse $\mathcal{T}^E$, reward model $\tilde{r}_{\boldsymbol{\Theta}}$, timesteps totali $T_{\mathrm{tot}}$, timesteps per iterazione $T$, gradient steps $G$, pre-warm steps $T_0$
>
> **Output:** Agente addestrato $\pi$

```
▷ Pre-warmup opzionale (una sola volta, prima del loop)
if T₀ > 0:
    TrainAgent(π, r̃_Θ, ℰ, T₀)                        # Algoritmo 5 — agente con reward corrente

▷ Loop principale
for iter = 0, 1, …, ⌊T_tot / T⌋ - 1:

    ▷ Fase 1 — Raccolta rollout
    𝒯^M ← SampleRollout(π, ℰ, T)                      # Algoritmo 2 — traiettorie fresche dalla policy

    ▷ Fase 2 — (no-op) Raccolta feedback
    # collect_feedback() restituisce ([], []) — nessuna query all'esperto
    # push_data() logga solo |𝒯^E| e |𝒯^M|

    ▷ Fase 3 — Validazione pre-training (log correlazione con ground truth)
    LogRewardCorrelation(r̃_Θ, 𝒯^M)                    # Kendall τ e MAE per stato terminale

    ▷ Fase 4 — Addestramento Reward Model
    TrainRewardModel(𝒯^E, 𝒯^M)                         # Algoritmo 3 — MaxEnt IRL

    ▷ Fase 5 — Normalizzazione RM (solo media)
    NormalizeRewardMean(r̃_Θ, 𝒯^M)                      # Algoritmo 4 — centra l'output in 0

    ▷ Fase 6 — Addestramento Agente
    TrainAgent(π, r̃_Θ, ℰ, T)                           # Algoritmo 5 — PPO con reward appreso

return π
```

### 5.2 Raccolta Rollout

> **Algorithm 2 — `SampleRollout` (raccolta traiettorie dall'agente)**
>
> **Input:** Agente $\pi$, env $\mathcal{E}$, passi $T$, passi esplorazione $T_{\varepsilon} = \varepsilon_{\mathrm{frac}} \cdot T$

```
𝒯^M ← esegui π su ℰ per ≥ T passi; raccogli episodi completi

if T_ε > 0:
    𝒯^M ← 𝒯^M ∪ RolloutRandom(π_ε, ℰ, T_ε)          # policy ε-greedy con ε = exploration_eps

for τ ∈ 𝒯^M:
    R_true(τ)  ← Σ_t  r_t^true
    R_model(τ) ← Σ_t  r̃_Θ.predict(o_t, a_t, s_{t+1}, d_t)

log(mean R_true, mean R_model, mean length, n_trajectories)

return 𝒯^M
```

### 5.3 Addestramento del Reward Model

> **Algorithm 3 — `TrainRewardModel` (MaxEnt IRL)**
>
> **Input:** $\mathcal{T}^E$ fisso, $\mathcal{T}^M$ corrente, ensemble $\{\mathrm{member}_k\}_{k=1}^K$, $G$ gradient steps, $n_e =$ `batch_size_expert`, $n_m =$ `batch_size_model`

```
for k = 1, …, K:                                      # parallelizzabile (1 thread per membro)
    member_k.train()

    for step = 1, …, G:

        ▷ Mini-batch esperto (senza rimessa)
        I_E ~ Uniform{0, …, |𝒯^E|-1}^{min(n_e, |𝒯^E|)}   # campionamento senza rimessa
        expert_returns ← [ R_k(τ^E_i) for i ∈ I_E ]
            where R_k(τ) = Σ_t  member_k(o_t, a_t, s_t, d_t)

        ▷ Mini-batch modello (senza rimessa, traiettorie fresche — no IS)
        I_M ~ Uniform{0, …, |𝒯^M|-1}^{min(n_m, |𝒯^M|)}
        model_returns  ← [ R_k(τ^M_j) for j ∈ I_M ]

        ▷ Perdita MaxEnt IRL
        log_Z ← logsumexp(model_returns) - log(|I_M|)
        ℒ_k  ← -mean(expert_returns) + log_Z

        ▷ Passo di discesa del gradiente (Adam + weight decay λ)
        θ_k ← θ_k - η · ∇_{θ_k} ℒ_k

▷ Valutazione (loss su snapshot corrente, con reward_model completo)
loss_val ← MaxEntLoss(reward_model, 𝒯^E, 𝒯^M)          # no_grad, eval mode
log(loss_val, time_train_reward_model)
```

In forma matematica, per ogni gradient step:

$$
\mathcal{L}_k = -\frac{1}{n_e}\sum_{i \in I_E} R_k(\tau^E_i) \;+\; \operatorname{logsumexp}_{j \in I_M}\bigl[R_k(\tau^M_j)\bigr] - \log n_m
$$

$$
\boldsymbol{\theta}_k \;\leftarrow\; \boldsymbol{\theta}_k - \eta\,\nabla_{\boldsymbol{\theta}_k}\mathcal{L}_k
$$

### 5.4 Normalizzazione della Media del Reward Model

> **Algorithm 4 — `NormalizeRewardMean` (centra l'output su 0)**
>
> **Input:** reward model $\tilde{r}_{\boldsymbol{\Theta}}$, transizioni correnti $\mathcal{T}^M$

```
transitions ← flatten(𝒯^M)                            # tutte le transizioni del rollout

▷ Normalizzazione di ogni membro interno (inner NormalizedRewardNet)
for k = 1, …, K:
    raw_k ← member_k.predict_unnormalized(obs, acts, status, done)
                                                       # output grezzo di SumoRewardNet_k
    μ_k ← mean(raw_k)
    member_k.set_mean(μ_k)                             # EMA: μ_k ← (1-α)μ_k + α·mean(raw_k)

▷ Normalizzazione dell'outer wrapper (outer NormalizedRewardNet)
raw_outer ← reward_model.predict_unnormalized(obs, acts, status, done)
                                                       # = mean_k(member_k.forward(...)) ≈ 0 dopo il passo sopra
μ_outer ← mean(raw_outer)
reward_model.set_mean(μ_outer)
```

In formula: dopo la normalizzazione, il reward model centrato è:

$$
\tilde{r}_{\boldsymbol{\Theta}}^{\mathrm{centered}}(o,a,\boldsymbol{s},d) = \frac{1}{K}\sum_k\bigl(r_{\boldsymbol{\theta}_k}(o,a,\boldsymbol{s},d) - \mu_k\bigr) - \mu_{\mathrm{outer}}
$$

con $\mu_k = \mathbb{E}_{t \in \mathcal{T}^M}\bigl[r_{\boldsymbol{\theta}_k}(o_t, a_t, \boldsymbol{s}_t, d_t)\bigr]$ e $\mu_{\mathrm{outer}} \approx 0$.

**Nota:** solo la **media** viene aggiornata in questa fase (non $\sigma$). La deviazione standard $\sigma$ rimane al suo valore iniziale (1.0), a meno che non venga aggiornata esplicitamente altrove.

### 5.5 Addestramento dell'Agente

> **Algorithm 5 — `TrainAgent` (PPO con reward appreso)**
>
> **Input:** Agente PPO $\pi$, reward model $\tilde{r}_{\boldsymbol{\Theta}}$, env $\mathcal{E}$, passi $T$

```
▷ L'env è wrappato: il reward vero è sostituito da r̃_Θ.predict(o, a, s, d)
π ← PPO.learn(ℰ_wrapped, steps=T, reward=r̃_Θ.predict)
```

Il reward che arriva a PPO per ogni transizione è:

$$
\hat{r}_t = \tilde{r}_{\boldsymbol{\Theta}}^{\mathrm{predict}}(o_t, a_t, \boldsymbol{s}_t, d_t) = \frac{\left[\frac{1}{K}\sum_k(r_{\boldsymbol{\theta}_k}(o_t, a_t, \boldsymbol{s}_t, d_t) - \mu_k)\right] - \mu_{\mathrm{outer}}}{\sigma_{\mathrm{outer}} + \varepsilon}
$$

---

## 6. Riepilogo dei Parametri

| Parametro | Default | Descrizione |
|---|---|---|
| `lr_rew` | $10^{-3}$ | Learning rate Adam $\eta$ per il RM |
| `gradient_steps_rew` | 10 | Gradient steps per iterazione ($G$) |
| `batch_size_expert` | 32 | Traiettorie esperto per mini-batch ($n_e$) |
| `batch_size_model` | 64 | Traiettorie modello per mini-batch ($n_m$) |
| `l2_rew` | $0.01$ | Weight decay (L2) per Adam |
| `fragment_length` | 1 | Ereditato dalla base class, non usato nella loss |
| `temperature` | 1.0 | Fattore di scala per la validazione (logging only) |
| `initial_agent_timesteps` | 0 | Timesteps di pre-warm dell'agente prima del loop |
| `exploration_frac` | 0.0 | Frazione di passi di esplorazione $\varepsilon$-greedy |
| `exploration_eps` | 0.5 | Epsilon per la policy di esplorazione |
| `query_schedule` | `"constant"` | Schedule delle query (no-op per DemoAlgorithm) |

---

## 7. Note Implementative e Osservazioni

**Differenza da ZhangAlgorithm (Bradley-Terry).** ZhangAlgorithm confronta frammenti dell'agente con frammenti dell'esperto sulle *stesse osservazioni* e usa una perdita Bradley-Terry. `DemoAlgorithm` invece lavora su **traiettorie complete**, usa dati esperto **fissi** e minimizza la perdita **MaxEnt IRL**. Non c'è nessuna query all'esperto durante il training: `collect_feedback` è un no-op.

**Nessun bootstrap / dataset split.** A differenza degli algoritmi preference-based, non si usa campionamento con rimessa né split train/val sul dataset: le traiettorie esperto sono campionate senza rimessa ad ogni gradient step, e le traiettorie modello sono sempre fresche dal rollout corrente (nessuna storia passata).

**Traiettorie modello fresche → no importance sampling.** Poiché $\mathcal{T}^M$ viene rigenerato ad ogni iterazione dalla policy corrente, il termine `logsumexp` è già un estimatore non biased della log-funzione di partizione $\log Z_{\boldsymbol{\theta}}$ senza necessità di correzioni IS.

**Double-wrapping del reward model.** Ogni membro dell'ensemble è un `NormalizedRewardNet(SumoRewardNet)` e l'intera catena è ulteriormente avvolta da un secondo `NormalizedRewardNet`. Questo permette di normalizzare separatamente ogni membro e poi l'ensemble complessivo.

**Validazione del RM.** Prima di ogni aggiornamento del RM, la base class logga metriche di correlazione con il ground-truth (MAE per tipo di stato terminale + Kendall $\tau$ sui passi `running`). Questo è indipendente dalla loss MaxEnt e serve esclusivamente per il monitoraggio.

**Pre-warm dell'agente.** Se `initial_agent_timesteps > 0`, l'agente viene allenato per $T_0$ passi con il reward model non ancora ottimizzato. Questo migliora la qualità di $\mathcal{T}^M$ alla prima iterazione, rendendo `logsumexp` un estimatore più accurato della funzione di partizione.

**Complessità per iterazione.** Il costo dominante è il training del RM: $O(K \cdot G \cdot (n_e + n_m) \cdot C_{\mathrm{fwd}})$, dove $C_{\mathrm{fwd}}$ è il costo di un forward pass per step. Il numero di gradient steps effettivi è esattamente $G = K \cdot G_\mathrm{member}$ (con i default $K=1,\; G=10$, si ottengono 10 gradient steps per iterazione).
