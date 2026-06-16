## 0. Overview del progetto

### 0.1 Cos'è il framework

Il pacchetto `human_feedback_rl` implementa **cinque algoritmi di reward/imitation learning** che condividono un **outer loop comune** (Template Method) definito in `common/base_reward_learning_algorithm.py::BaseRewardLearningAlgorithm.train`. L'ambiente target è un task di guida autonoma SUMO/Highway (`HighwayEgo-v0`, ego continuo, reward `fast`), dove l'osservazione è continua, l'azione è continua e ogni transizione porta uno `next_status` one-hot a 7 dimensioni `[arrived, collided, off_road, timeout, running, teleported, removed_unknown]` (`common/env_wrappers.py::_STATUS_ONEHOT`).

Il loop alterna due fasi a ogni iterazione (`base_reward_learning_algorithm.py:430-465`):

1. **Rollout**: si raccolgono `timesteps_per_iteration` transizioni con la policy corrente (`sample_rollout`).
2. **Feedback** (solo se l'algoritmo lo usa): si frammentano le traiettorie e si raccolgono etichette (`collect_feedback` → `push_data`).
3. **Training del reward model** (`train_reward_model`).
4. **Training della policy** con PPO usando il reward appreso (`train_agent`), dove il reward del modello sostituisce quello dell'ambiente tramite `EnvRewardWrapper`.

PPO è l'algoritmo di RL "outer" in tutti i casi tranne DAgger (che fa pura behaviour cloning supervisionata).

### 0.2 Mappatura algoritmo → script → paper (verificata)

| Script | Config | Classe | Paper di riferimento | Tipo di supervisione |
|---|---|---|---|---|
| `scripts/test_chri_PPO.py` | `configs/test_chri_PPO.yaml` | `PreferenceAlgorithm` | Christiano et al. 2017 (*Deep RL from Human Preferences*) | **Preferenze** pairwise su coppie di frammenti |
| `scripts/test_demo_PPO.py` | `configs/test_demo_PPO.yaml` | `DemoAlgorithm` | Finn et al. 2016 (*Guided Cost Learning*) | **Dimostrazioni** esperte (MaxEnt IRL) |
| `scripts/test_gail_PPO.py` | `configs/test_gail_PPO.yaml` | `GailAlgorithm` | Ho & Ermon 2016 (*GAIL*) | **Dimostrazioni** (imitation avversariale) |
| `scripts/test_airl_PPO.py` | `configs/test_airl_PPO.yaml` | `AirlAlgorithm` | Fu et al. 2018 (*AIRL*) | **Dimostrazioni** (IRL avversariale) |
| `scripts/eval.py` (no config dedicato `test_*`) | — | `DaggerAlgorithm` | Ross et al. 2011 (*DAgger*) | **Dimostrazioni interattive** (expert queryabile) |

La gerarchia di ereditarietà è significativa:

```
BaseAlgorithm
 └─ BaseRewardLearningAlgorithm  (outer loop comune, PPO, reward-correlation logging)
     ├─ PreferenceAlgorithm      (Bradley-Terry su preferenze)
     └─ DemoAlgorithm            (MaxEnt IRL / Guided Cost Learning)
         └─ GailAlgorithm        (sostituisce la loss IRL con BCE del discriminatore)
             └─ AirlAlgorithm    (discriminatore con reward+value shaping)
BaseAlgorithm
 └─ DaggerAlgorithm              (NON eredita dal reward-learning; è imitation supervisionata)
```

> **Nota strutturale importante per la discussione.** GAIL e AIRL ereditano da `DemoAlgorithm` ma **non usano la loss MaxEnt IRL del padre**: la sovrascrivono in `_compute_reward_loss` e annullano il calcolo dei pesi di importance sampling (`_update_importance_weights` → `pass`). L'ereditarietà serve solo a riusare il meccanismo di buffering delle dimostrazioni e l'outer loop. Questo è un punto su cui il relatore potrebbe insistere (vedi §8).

### 0.3 Relazione tra preferenze, dimostrazioni, reward learning e policy learning

- **Reward learning vs policy learning.** Quattro algoritmi su cinque sono *reward learning*: imparano una funzione di reward $R_\theta$ (o un discriminatore da cui si deriva un reward) e poi addestrano la policy con PPO su quel reward. DAgger invece è *policy learning diretto*: copia le azioni dell'esperto senza mai stimare un reward.
- **Preferenze vs dimostrazioni.** Sono due forme di feedback complementari:
  - *Preferenze* (Christiano): l'oracolo ordina due comportamenti ($\tau^1 \succ \tau^2$). Non richiede un esperto in grado di agire, solo di giudicare. Modello statistico: Bradley-Terry.
  - *Dimostrazioni* (Demo/GAIL/AIRL/DAgger): l'oracolo fornisce traiettorie/azioni esperte. Modello statistico: massima entropia (Demo/AIRL) o matching di distribuzione di occupazione (GAIL).
- **Punto di contatto teorico.** Sia il modello Bradley-Terry sulle preferenze sia il modello MaxEnt sulle dimostrazioni assumono che la probabilità di un comportamento sia **esponenziale nel suo reward cumulato**: $p(\tau) \propto \exp(R_\theta(\tau))$. È la stessa ipotesi di "Boltzmann-rationality". Christiano la applica al *rapporto* tra due traiettorie; MaxEnt IRL la applica alla *densità* assoluta di una traiettoria (da cui la funzione di partizione $Z$). AIRL chiude il cerchio: il suo discriminatore ottimale recupera esattamente il reward MaxEnt.

### 0.4 Pseudo-algoritmo dell'outer loop comune

Tutti gli algoritmi reward-based (Preference, Demo, GAIL, AIRL) condividono il loop seguente (`base_reward_learning_algorithm.py::train`, `:407-468`). I passi specifici di ciascun algoritmo vivono in `collect_feedback`, `push_data`, `train_reward_model` e `before_agent_training`, sovrascritti nelle sottoclassi.

```
Algoritmo 0 — Outer loop comune (BaseRewardLearningAlgorithm.train)
Input: env, policy π (PPO), reward model R_θ, oracolo di feedback
n_iterations = total_timesteps / timesteps_per_iteration
ripeti per n_iterations:
  1. ROLLOUT   : raccogli timesteps_per_iteration transizioni con π  → self.trajectories
  2. FEEDBACK  : (solo se l'algoritmo lo usa)
       frammenti = fragmenter(self.trajectories)
       etichette = collect_feedback(frammenti)      # oracolo sintetico
       push_data(frammenti, etichette)              # aggiorna dataset / buffer / pesi IS
  3. REWARD    : train_reward_model()               # gradient_steps_rew passi di SGD
  4. PREP      : before_agent_training()            # set normalizzazione / centratura logit
  5. POLICY    : train_agent()                      # PPO su R_θ via EnvRewardWrapper
  6. LOG       : correlazione reward appreso vs vero (Kendall / MAE)
```

---

## 1. PreferenceAlgorithm — RL from Human Preferences (Christiano et al. 2017)

**File:** `algorithms/preference_algorithm.py` · **Paper:** *Deep RL from Human Preferences*.

### 1.1 Obiettivo

Non disponiamo del reward vero. Disponiamo di un oracolo che, date due brevi traiettorie (frammenti) $\sigma^1, \sigma^2$, dice quale preferisce. Vogliamo imparare $\hat r_\theta(s,a)$ tale che le preferenze indotte da $\hat r_\theta$ coincidano con quelle dell'oracolo, e poi ottimizzare la policy su $\hat r_\theta$.

### 1.2 Formulazione matematica

**Dataset.** Coppie $\big(\sigma^1, \sigma^2, \mu\big)$ dove $\sigma^i = \big((s^i_0,a^i_0),\dots,(s^i_{k-1},a^i_{k-1})\big)$ è un frammento di lunghezza $k$ e $\mu = (\mu_1,\mu_2)$ è la distribuzione di preferenza (one-hot $(1,0)$/$(0,1)$, oppure $(0.5,0.5)$ per indifferenza).

**Modello di preferenza (Bradley-Terry / Luce-Shephard).** Si assume che la probabilità di preferire $\sigma^1$ sia funzione logistica della differenza dei reward cumulati predetti:

$$
\hat P[\sigma^1 \succ \sigma^2] \;=\; \frac{\exp\!\sum_t \hat r_\theta(s^1_t,a^1_t)}{\exp\!\sum_t \hat r_\theta(s^1_t,a^1_t) + \exp\!\sum_t \hat r_\theta(s^2_t,a^2_t)} \;=\; \sigma\!\Big(\textstyle\sum_t \hat r_\theta(s^1_t,a^1_t) - \sum_t \hat r_\theta(s^2_t,a^2_t)\Big),
$$

dove $\sigma$ è la sigmoide logistica.

**Loss (cross-entropy / negative log-likelihood).** Eq. (1) del paper:

$$
\mathcal L(\theta) \;=\; -\sum_{(\sigma^1,\sigma^2,\mu)} \Big[\mu_1 \log \hat P[\sigma^1\succ\sigma^2] + \mu_2 \log \hat P[\sigma^2\succ\sigma^1]\Big].
$$

È la log-likelihood negativa del modello Bradley-Terry; minimizzarla equivale a stimare $\theta$ per massima verosimiglianza sulle preferenze osservate.

**Policy.** $\pi$ è addestrata con PPO massimizzando $\mathbb E_{\pi}\big[\sum_t \hat r_\theta(s_t,a_t)\big]$, col reward normalizzato a media 0 / std 1 (paper §2.2).

### 1.3 Implementazione

- **Reward model:** ensemble di `NormalizedRewardNet(SumoRewardNet)` costruito da `make_reward_ensemble` (`reward_nets.py:340`), a sua volta avvolto in un `NormalizedRewardNet` esterno (doppia normalizzazione, vedi §1.6). Default config: `n_ensembles: 3`, `net_arch: [32,32]`, `tanh`.
- **Raccolta feedback:** `collect_feedback` (`preference_algorithm.py:87`) → il fragmenter produce coppie, il `PreferenceGathererFromReward` (oracolo sintetico) le etichetta.
- **Oracolo sintetico:** `gatherers.py::PreferenceGathererFromReward` usa il reward **vero** della traiettoria per generare le preferenze (modalità `binary`, `soft`, `binary_bernulli`). Config usa `binary_bernulli` con `temperature: 20`.
- **Training reward:** `train_reward_model` (`preference_algorithm.py:124`). Per ogni membro si fa **bootstrap** del dataset (`dataset.bootstrap()`), poi `gradient_steps_rew` (=100) passi di SGD.
- **Active learning:** `fragmenter_type: "active"` → `HighVariancePairFragmenter` seleziona le coppie su cui l'ensemble è più in disaccordo (varianza del return predetto), §3 del paper.

**Loss nel codice** (`preference_algorithm.py:133-143`):

```python
r1 = th.stack([member.fragment_avg_reward(p.frag1) for p in batch.fragment_pairs])
r2 = th.stack([member.fragment_avg_reward(p.frag2) for p in batch.fragment_pairs])
prob1 = th.sigmoid(r1 - r2)
bt_probs = th.stack([prob1, 1 - prob1], dim=1)
labels = th.tensor([[p.pref1, p.pref2] for p in batch.preferences])
loss = -(labels * bt_probs.clamp(min=1e-7).log()).sum(dim=1).mean()
```

**Pseudo-algoritmo dell'implementazione:**

```
Algoritmo 1 — PreferenceAlgorithm (Christiano)
collect_feedback(self.trajectories):
  coppie (σ1,σ2) ← fragmenter            # "active" → HighVariancePairFragmenter
  per ogni coppia:
    μ ← PreferenceGathererFromReward(σ1,σ2)   # p1 = σ((R_true(σ1)-R_true(σ2))/T), T=20
                                              # label "binary_bernulli": μ ~ Bernoulli(p1)
  push_data → dataset di coppie etichettate

train_reward_model():
  per ogni membro m dell'ensemble (sequenziale):
    D_m ← dataset.bootstrap()              # campionamento con rimpiazzo
    ripeti gradient_steps_rew (=100) volte:
      batch ← minibatch(D_m)
      r1 = m.fragment_avg_reward(frag1);  r2 = m.fragment_avg_reward(frag2)
      p1 = σ(r1 - r2);   bt = [p1, 1-p1]                      # T = 1 nel modello
      loss = -(labels · log clamp(bt, 1e-7)).sum().mean()     # NLL Bradley-Terry
      θ_m ← θ_m - lr_rew · ∇loss
before_agent_training(): set _mean dell'ensemble per la normalizzazione PPO
reward per PPO: r̄_θ(s,a) normalizzato (media 0 / std 1) in EnvRewardWrapper
```

### 1.4 Loss function — confronto formula/codice

| Termine teorico | Codice |
|---|---|
| $\sum_t \hat r_\theta(s^1_t,a^1_t)$ | `member.fragment_avg_reward(p.frag1)` — **MEDIA**, non somma (vedi §1.6) |
| $\sigma(R^1 - R^2)$ | `th.sigmoid(r1 - r2)` ✓ |
| $\hat P = [\sigma, 1-\sigma]$ | `bt_probs = stack([prob1, 1-prob1])` ✓ |
| $-\sum \mu_i \log \hat P_i$ | `-(labels * bt_probs.clamp(1e-7).log()).sum(dim=1).mean()` ✓ |

Il segno è corretto: si **minimizza** la cross-entropy (NLL). Il `clamp(min=1e-7)` evita $\log 0$. La media (`.mean()`) sul batch è una normalizzazione standard innocua.

### 1.5 Mappatura teoria → codice

| Concetto teorico | Implementazione |
|---|---|
| Frammento $\sigma$ | `types.Fragment` (= `Trajectory`) |
| Preferenza $\mu$ | `types.Preference(pref1, pref2)` con vincolo $\mu_1+\mu_2=1$ |
| Reward cumulato $R^i$ | `fragment_avg_reward` (media per-step) |
| $\hat P[\sigma^1\succ\sigma^2]$ | `th.sigmoid(r1 - r2)` |
| Loss Eq.(1) | riga 143 |
| Ensemble + bootstrap | `reward_model.members` + `dataset_train.bootstrap()` |
| Active query (disaccordo) | `HighVariancePairFragmenter` |
| Reward normalizzato per PPO | `EnvRewardWrapper.step_wait` (running mean/std) |

### 1.6 Verifica di correttezza e problemi

1. **Media invece di somma nel reward del frammento (concettuale).** Il paper usa $\sum_t \hat r$; il codice usa $\frac1k\sum_t \hat r$ (`fragment_avg_reward`, `reward_nets.py:95-101`). **Con `fragment_length: 1` (config attuale) i due coincidono** perché $k=1$. Per $k>1$ differirebbero solo per un fattore costante $1/k$ **se i frammenti hanno lunghezza uguale**; se le lunghezze differiscono, media e somma non sono più proporzionali e cambia il modello. Da segnalare ma **non è un bug nella configurazione usata**.

2. **Coerenza oracolo↔modello sulla scala del reward.** L'oracolo `soft`/`binary_bernulli` genera le etichette con $\text{prob}_1 = 1/(1+e^{(r_2-r_1)/T})$, $T=20$ (`gatherers.py:38,42`), mentre il modello fitta con $\sigma(r_1-r_2)$ (cioè $T=1$). Non è un errore di correttezza: la scala del reward è arbitraria (verrà normalizzata), quindi il modello imparerà semplicemente reward riscalati. Con `binary_bernulli` l'effetto pratico di $T=20$ è **rumore di etichetta** (le preferenze diventano stocastiche quando $|r_1-r_2|$ è piccolo rispetto a 20). Buono per realismo, ma da dichiarare.

3. **Frammenti di lunghezza 1 (concettuale).** Con `fragment_length: 1` la preferenza è su singole transizioni: il modello Bradley-Terry degenera a un confronto di reward per-step. Si perde la struttura temporale che è il punto di forza del confronto su segmenti nel paper. Scelta legittima per un reward markoviano, ma è una semplificazione rispetto a Christiano (che usa segmenti di ~1-2 s).

4. **Doppia normalizzazione del reward net (implementativo).** `make_reward_ensemble` avvolge ogni membro in `NormalizedRewardNet` **e** avvolge l'ensemble in un secondo `NormalizedRewardNet` (`reward_nets.py:351-361`). In `forward` (training) la normalizzazione usa `_mean=0,_std=1` finché non si chiama `set_mean/set_std`, quindi non altera la loss; `before_agent_training` setta solo `_mean`. Non è un bug, ma è un wrapping ridondante e fragile (due livelli di stato di normalizzazione da tenere coerenti).

5. **"Parallel" training fittizio (implementativo).** `ThreadPoolExecutor(max_workers=1)` (`preference_algorithm.py:150`): nonostante il commento "Train all members in parallel", `max_workers=1` rende il training **sequenziale** (e il GIL lo renderebbe comunque tale). Nessun impatto sulla correttezza, solo sul commento.

6. **Accuracy con pareggi (minore).** In `_evaluate_reward_model` (`:189`), `bt_probs.argmax == labels.argmax`: per etichette $(0.5,0.5)$ `argmax` ritorna 0, distorcendo l'accuracy sulle indifferenze. Metrica di logging, nessun effetto sul training.

7. **Normalizzazione reward per PPO non resetta (implementativo, condiviso).** `EnvRewardWrapper._reward_stats` è una running mean/std **cumulativa su tutto il training** (Welford, `env_wrappers.py:23-43,87-88`). Quando il reward model cambia tra iterazioni, la normalizzazione si adatta lentamente (la media è dominata dalla storia). Possibile sorgente di non-stazionarietà del segnale PPO. Vale per tutti gli algoritmi reward-based.

**Verdetto:** l'implementazione Bradley-Terry è **fedele e corretta** (segni, sigmoide, cross-entropy, ensemble+bootstrap, active query). Le divergenze dal paper sono scelte di semplificazione (frammenti lunghezza 1, media vs somma) coerenti con la config, non errori.

### 1.7 Il ruolo della temperatura nella loss basata su preferenze

Questa sezione è centrale per la discussione: spiega **perché** la differenza di reward $\Delta = r_\theta(\tau_w) - r_\theta(\tau_l)$ non andrebbe usata "nuda" dentro la sigmoide, e in che senso la temperatura $T$ (o $\beta = 1/T$) rende l'ottimizzazione del reward model più stabile. Distinguo sempre la **motivazione teorica** dalla **collocazione effettiva nel codice** (§1.7.7), che è un punto su cui il relatore può incalzare.

#### 1.7.1 Il problema: usare direttamente $\Delta = r_\theta(\tau_w)-r_\theta(\tau_l)$

La probabilità Bradley-Terry per la coppia (preferito $w$, scartato $l$) è $P(\tau_w \succ \tau_l) = \sigma(\Delta)$, e per un'etichetta hard $\mu=(1,0)$ la loss per esempio è la NLL

$$
\ell(\theta) = -\log \sigma(\Delta), \qquad \Delta = r_\theta(\tau_w) - r_\theta(\tau_l).
$$

Il reward $r_\theta$ ha **scala arbitraria e non vincolata**: la rete può produrre output di ampiezza qualunque. Ne derivano due patologie accoppiate.

1. **Indeterminatezza di scala / minimo all'infinito.** Su un dataset di preferenze *linearmente separabile* (tutte le coppie ordinabili coerentemente), per qualunque $\theta$ che ordina correttamente le coppie, riscalare $r_\theta \mapsto c\,r_\theta$ con $c>1$ aumenta ogni $\Delta>0$ e quindi **riduce monotonamente** la loss: $-\log\sigma(c\Delta) \to 0$ per $c\to\infty$. La loss BT **non ha un minimo finito**: l'ottimizzatore è spinto a far divergere $\lVert\theta\rVert$ e a saturare le sigmoidi. Il reward "vince" l'obiettivo gonfiando la propria scala invece di migliorare l'ordinamento — una forma di reward hacking dell'obiettivo di preferenza.

2. **Argomento della sigmoide fuori scala.** Anche prima della divergenza, se la scala tipica di $\Delta$ è grande rispetto a 1 (la larghezza naturale della sigmoide), l'argomento finisce nelle **code** di $\sigma$, dove la funzione è piatta. Lì il segnale di gradiente svanisce (§1.7.3) e l'apprendimento si blocca per le coppie già ordinate, anche se l'ordinamento non è ancora robusto.

#### 1.7.2 Saturazione della sigmoide

La logistica $\sigma(x) = \dfrac{1}{1+e^{-x}}$ ha tre regimi:

- $x \approx 0$: regione **lineare/ad alta curvatura**, $\sigma(0)=\tfrac12$, pendenza massima $\sigma'(0)=\tfrac14$;
- $x \to +\infty$: $\sigma(x)\to 1$ esponenzialmente, $1-\sigma(x)\approx e^{-x}$;
- $x \to -\infty$: $\sigma(x)\to 0$, $\sigma(x)\approx e^{x}$.

Per $|x|$ grande la funzione è **satura**: variazioni anche ampie di $x$ producono variazioni infinitesime di $\sigma(x)$. Numericamente compaiono anche underflow/overflow (mitigati nel codice da `clamp(min=1e-7)` prima del `log`, `preference_algorithm.py:143`, e dall'uso implicito di un'aritmetica stabile).

#### 1.7.3 Derivata della sigmoide e gradiente che svanisce

Derivata in forma chiusa:

$$
\sigma'(x) = \frac{e^{-x}}{(1+e^{-x})^2} = \sigma(x)\big(1-\sigma(x)\big).
$$

Questo prodotto è **massimo in $x=0$** (vale $\tfrac14$) e tende a $0$ per $|x|\to\infty$, perché uno dei due fattori $\sigma$ o $1-\sigma$ collassa esponenzialmente:

$$
\sigma'(x) \;\xrightarrow{\,x\to+\infty\,}\; (1)\cdot e^{-x} \to 0, \qquad
\sigma'(x) \;\xrightarrow{\,x\to-\infty\,}\; e^{x}\cdot(1) \to 0.
$$

**Gradiente della loss.** Per $\ell = -\log\sigma(\Delta)$:

$$
\frac{\partial \ell}{\partial \Delta} = -\frac{\sigma'(\Delta)}{\sigma(\Delta)} = -\big(1-\sigma(\Delta)\big) = \sigma(\Delta)-1 \in (-1,0),
$$

e per la regola della catena rispetto ai parametri:

$$
\nabla_\theta \ell = \big(\sigma(\Delta)-1\big)\,\nabla_\theta \Delta = -\big(1-\sigma(\Delta)\big)\,\big(\nabla_\theta r_\theta(\tau_w) - \nabla_\theta r_\theta(\tau_l)\big).
$$

Il **modulo** del segnale è $1-\sigma(\Delta)$:

- coppia ordinata con **alta confidenza** ($\Delta \gg 0$): $1-\sigma(\Delta)\to 0$ ⇒ **gradiente nullo**. Per esempi già corretti va bene, ma se la scala è gonfiata questo congela prematuramente *anche* coppie il cui margine non è ancora affidabile;
- coppia **confidentemente errata** ($\Delta \ll 0$): $1-\sigma(\Delta)\to 1$ ⇒ il segnale satura a modulo $1$ (limitato: la BT non esplode), ma se $\nabla_\theta\Delta$ è piccolo per saturazione interna della rete il passo effettivo resta minuscolo;
- $\Delta \approx 0$: $1-\sigma\approx\tfrac12$ ⇒ **gradiente massimo e informativo**.

Conclusione: l'apprendimento è efficace **solo finché $\Delta$ resta nella banda $O(1)$ attorno a 0**. Se la scala del reward cresce, quasi tutte le coppie finiscono nelle code, $\nabla_\theta\ell \to 0$, e il training stalla pur essendo la loss lontana dall'ottimo "robusto".

#### 1.7.4 Perché si introduce la temperatura $T$ (o $\beta = 1/T$)

Si riscala l'argomento della sigmoide con una temperatura $T>0$:

$$
\boxed{\;P(\tau_w \succ \tau_l) = \sigma\!\left(\frac{r_\theta(\tau_w)-r_\theta(\tau_l)}{T}\right) = \sigma\!\big(\beta\,\Delta\big), \qquad \beta = \frac1T.\;}
$$

La temperatura **disaccoppia la scala del reward dalla scala dell'argomento della sigmoide**. Scegliendo $T$ dell'ordine dell'ampiezza tipica di $\Delta$, l'argomento $\Delta/T$ resta $O(1)$, cioè nella regione ad alta curvatura dove $\sigma'$ è massima e i gradienti sono informativi. La temperatura agisce quindi come **normalizzatore di scala dell'obiettivo**, complementare (o alternativo) a weight decay e normalizzazione del reward.

#### 1.7.5 Effetto di $T$ su forma della sigmoide, gradienti e stabilità

Il gradiente con temperatura diventa

$$
\frac{\partial \ell}{\partial \theta} = -\frac{1}{T}\big(1-\sigma(\Delta/T)\big)\,\nabla_\theta \Delta .
$$

Due effetti contrapposti, da bilanciare:

- **$T$ piccolo ($\beta$ grande) ⇒ sigmoide ripida**, vicina a un gradino. Il fattore $1/T$ amplifica il gradiente, **ma** $\sigma(\Delta/T)$ satura per $|\Delta|$ minuscoli: il gradiente è grande solo in una banda strettissima attorno a $\Delta=0$ e **nullo ovunque altrove**. Risultato: training **instabile**, sensibile al rumore di etichetta, con la maggior parte delle coppie "morte".
- **$T$ grande ($\beta$ piccolo) ⇒ sigmoide piatta**. $\sigma(\Delta/T)\approx \tfrac12 + \tfrac{\Delta}{4T}$: la loss è quasi lineare in $\Delta$, gradienti **piccoli ma non saturi e ben condizionati** su un ampio range. Training **stabile ma lento**, e con bias verso la non-discriminazione.
- **$T$ intermedio**: mantiene $\Delta/T = O(1)$ per i $\Delta$ tipici del dataset ⇒ massimizza il numero di coppie nella regione informativa della sigmoide. È il regime desiderato.

Matematicamente, perché la temperatura *migliora* l'ottimizzazione: il modulo medio del gradiente sul dataset è $\mathbb{E}[\,1-\sigma(\Delta/T)\,]/T$. Per $T\to 0$ esso è non nullo solo su un insieme di misura evanescente (banda $|\Delta|\lesssim T$); per $T$ adeguato la frazione di coppie con $\sigma'$ vicino al massimo è massimizzata, dando una **norma del gradiente più uniforme e meglio condizionata** (Hessiana meno degenere) ⇒ discesa più rapida e stabile. Inoltre, fissare $T$ (o equivalentemente normalizzare/regolarizzare la scala) **ripristina un minimo finito**: senza riscalamento la loss tende a $0$ solo all'infinito (§1.7.1), mentre vincolando la scala dell'argomento l'ottimo cade a $\lVert\theta\rVert$ finito.

#### 1.7.6 Interpretazione: temperatura = razionalità/confidenza del modello BTL

Nel modello **Bradley-Terry-Luce** la probabilità di scelta è $P(i \succ j) = \dfrac{e^{\beta s_i}}{e^{\beta s_i}+e^{\beta s_j}} = \sigma\big(\beta(s_i-s_j)\big)$, dove $\beta=1/T$ è il parametro di **discriminabilità/razionalità** (equivalente alla temperatura di Boltzmann nel modello di scelta razionale-rumorosa di Luce/Ziebart):

- $\beta \to \infty$ ($T\to 0$): scelta **deterministica e perfettamente razionale** — chi sceglie prende sempre l'opzione con reward maggiore (gradino di Heaviside);
- $\beta \to 0$ ($T\to\infty$): scelta **casuale** — indifferenza ($P\to\tfrac12$), valutatore non informativo;
- $\beta$ finito: valutatore **rumorosamente razionale** — preferisce il migliore ma con probabilità crescente di errore al ridursi del margine $|s_i-s_j|$.

La temperatura quindi codifica **quanta confidenza/coerenza si attribuisce all'oracolo di preferenza**. È esattamente l'ipotesi di Boltzmann-rationality che unifica Christiano, MaxEnt IRL e AIRL (§0.3): la stessa $\beta$ che qui scala l'argomento della sigmoide è il coefficiente dell'esponenziale nel modello $p(\tau)\propto e^{\beta R_\theta(\tau)}$.

#### 1.7.7 Collegamento teoria → codice (dove vive davvero la temperatura)

Punto da dichiarare con onestà al relatore: **in questa implementazione la temperatura NON compare nella loss di training del reward model**. La BT loss è calcolata con argomento "nudo" $\sigma(r_1 - r_2)$, cioè $T=1$ implicito (`preference_algorithm.py:136`):

```python
prob1 = th.sigmoid(r1 - r2)          # nessuna divisione per T  →  T = 1
```

La temperatura `temperature` (config Chri: `20.0`) è invece usata in **due punti diversi**:

1. **Oracolo sintetico di preferenza** — `gatherers.py::PreferenceGathererFromReward` (`:38`, `:42`):

   ```python
   prob1 = 1.0 / (1.0 + math.exp((r2 - r1) / self.temperature))   # = σ((r1 - r2)/T)
   ```

   Qui $T$ è il parametro **BTL del generatore di etichette**: modella la razionalità dell'oracolo sul reward *vero*. Con `labels_type: "binary_bernulli"` le etichette sono campionate da questa $P$, quindi $T=20$ inietta **rumore di preferenza** controllato (coppie con margine $\ll 20$ ricevono etichette quasi casuali). È l'interpretazione §1.7.6 applicata al *lato dati*, non al *lato modello*.

2. **Normalizzazione per il logging** — `base_reward_learning_algorithm.py::_normalize_predictions` (`:288`): `pred_rewards = pred_rewards * self.temperature`, solo per allineare le predizioni al reward vero nelle metriche MAE/Kendall, senza effetto sul training.

**Quale problema pratico risolve qui la temperatura, e cosa controlla la scala nella loss.** Dato che la loss del modello usa $T=1$, l'indeterminatezza di scala (§1.7.1) **non** è gestita dalla temperatura ma dovrebbe esserlo da: (a) il **weight decay** `l2_rew` e (b) la **normalizzazione** `NormalizedRewardNet` + la normalizzazione del reward in `EnvRewardWrapper`. **Criticità da segnalare:** nel config Christiano `l2_rew: 0.0` ⇒ la regolarizzazione di scala via L2 è **disattivata**. In assenza sia di temperatura nella loss sia di L2, l'unico freno alla saturazione/divergenza della scala del reward sono il numero finito di `gradient_steps_rew` e la normalizzazione a valle. Per la difesa, due risposte coerenti col framework teorico sopra:

- **mitigazione minima**: riattivare `l2_rew > 0` (penalizza $\lVert\theta\rVert$, impedendo il minimo all'infinito e tenendo $\Delta$ in banda);
- **mitigazione esplicita**: introdurre la temperatura *anche* nella loss del modello, $\sigma\big((r_1-r_2)/T\big)$, fissando $T$ alla scala tipica di $\Delta$ per mantenere i gradienti nella regione ad alta curvatura (§1.7.5).

In sintesi: la temperatura nel codice è oggi un parametro del **modello di preferenza che genera i dati** (razionalità dell'oracolo), mentre la sua funzione teorica di **stabilizzatore dei gradienti nella loss del reward model** non è sfruttata — ed è proprio questa la motivazione matematica (§1.7.1–1.7.5) che giustifica perché *si dovrebbe* introdurla.

---

## 2. DemoAlgorithm — MaxEnt IRL / Guided Cost Learning (Finn et al. 2016)

**File:** `algorithms/demo_algorithm.py` · **Paper:** *Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization*.

### 2.1 Obiettivo

Date solo dimostrazioni esperte $\{\tau^E\}$, imparare un reward $R_\theta$ sotto cui le dimostrazioni siano (quasi) ottime, nel senso del **principio di massima entropia**: l'esperto campiona traiettorie con probabilità esponenziale nel reward.

### 2.2 Formulazione matematica

**Modello MaxEnt (Ziebart 2008).** L'esperto genera traiettorie secondo

$$
p_\theta(\tau) \;=\; \frac{1}{Z(\theta)} \exp\big(R_\theta(\tau)\big), \qquad R_\theta(\tau)=\sum_t r_\theta(s_t,a_t), \qquad Z(\theta)=\int \exp\big(R_\theta(\tau)\big)\, d\tau .
$$

Qui $p_\theta(\tau)$ è la **distribuzione di probabilità sulle traiettorie** indotta dal reward $R_\theta$: è la densità con cui un agente che massimizza il reward atteso *sotto vincolo di massima entropia* genererebbe la traiettoria $\tau=(s_0,a_0,s_1,a_1,\dots)$. La forma è una **distribuzione di Boltzmann/Gibbs** sul reward cumulato $R_\theta(\tau)$: traiettorie a reward più alto sono esponenzialmente più probabili, ma tutte hanno probabilità non nulla (l'entropia evita il collasso sulla sola traiettoria ottima, modellando la sub-ottimalità dell'esperto). Il fattore $Z(\theta)$ (la **funzione di partizione**) è la costante di normalizzazione che integra il numeratore su tutte le traiettorie ammissibili affinché $\int p_\theta(\tau)\,d\tau = 1$; dipende da $\theta$ ed è il termine intrattabile dell'intero problema.

**Massima verosimiglianza.** Si massimizza la log-likelihood delle dimostrazioni, ovvero si minimizza:

$$
\mathcal L(\theta) \;=\; -\frac1N\sum_{i=1}^N R_\theta(\tau^E_i) \;+\; \log Z(\theta).
$$

Il primo termine alza il reward sulle dimostrazioni; il secondo (la **log-partition function**) lo abbassa ovunque.

**Calcolo del gradiente.** Il gradiente del primo termine è immediato (la somma è lineare in $R_\theta$):

$$
\nabla_\theta\Big(-\frac1N\sum_{i=1}^N R_\theta(\tau^E_i)\Big) = -\frac1N\sum_{i=1}^N \nabla_\theta R_\theta(\tau^E_i) = -\,\mathbb E_{\tau^E}\big[\nabla_\theta R_\theta\big].
$$

Per il secondo termine si usa la regola della catena su $\log Z(\theta)$ e si porta il gradiente dentro l'integrale:

$$
\nabla_\theta \log Z(\theta) = \frac{\nabla_\theta Z(\theta)}{Z(\theta)} = \frac1{Z(\theta)}\,\nabla_\theta\!\int \exp\big(R_\theta(\tau)\big)\, d\tau = \frac1{Z(\theta)}\int \nabla_\theta \exp\big(R_\theta(\tau)\big)\, d\tau .
$$

Usando $\nabla_\theta \exp(R_\theta) = \exp(R_\theta)\,\nabla_\theta R_\theta$ e riconoscendo $\tfrac{1}{Z(\theta)}\exp(R_\theta(\tau)) = p_\theta(\tau)$:

$$
\nabla_\theta \log Z(\theta) = \int \frac{\exp\big(R_\theta(\tau)\big)}{Z(\theta)}\,\nabla_\theta R_\theta(\tau)\, d\tau = \int p_\theta(\tau)\,\nabla_\theta R_\theta(\tau)\, d\tau = \mathbb E_{\tau\sim p_\theta}\big[\nabla_\theta R_\theta\big].
$$

Sommando i due contributi:

$$
\nabla_\theta \mathcal L(\theta) = -\,\mathbb E_{\tau^E}\big[\nabla_\theta R_\theta\big] + \mathbb E_{\tau\sim p_\theta}\big[\nabla_\theta R_\theta\big].
$$

All'ottimo i due termini si bilanciano ($\mathbb E_{\tau^E}[\nabla R_\theta] = \mathbb E_{\tau\sim p_\theta}[\nabla R_\theta]$): il reward è stazionario quando le **feature attese** sotto la distribuzione del modello eguagliano quelle delle dimostrazioni (*feature matching*, Ziebart 2008). L'intrattabilità di $\mathbb E_{\tau\sim p_\theta}$ — equivalente a quella di $Z(\theta)$ — è ciò che il punto seguente risolve via importance sampling.

**Stima di $Z$ via importance sampling (il contributo chiave di GCL).** $Z$ è intrattabile. Si stima con campioni generati da una distribuzione di background $q$ (la policy, o una **mistura** di policy passate). Con $M$ campioni:

$$
Z(\theta) \approx \frac1M \sum_{j=1}^M \frac{\exp\big(R_\theta(\tau_j)\big)}{q(\tau_j)}, \qquad
\log Z(\theta) \approx \operatorname{logsumexp}_j\Big(R_\theta(\tau_j) + \underbrace{\log z_j}_{-\log q(\tau_j)}\Big) - \log M .
$$

**Distribuzione di fusione (Finn et al. 2016, §4.3).** Per ridurre la varianza, $q$ è la **mistura** delle ultime $W$ policy che hanno generato i campioni:

$$
q_{\text{mix}}(\tau) = \frac1W\sum_{\kappa=1}^W q_\kappa(\tau), \qquad
\log z_j = -\log q_{\text{mix}}(\tau_j) = \log W - \operatorname{logsumexp}_\kappa \Big(\sum_t \log\pi_\kappa(a_t\mid s_t)\Big).
$$

**Loss finale implementata** (Algorithm 2, con le demo aggiunte all'insieme dei campioni):

$$
\boxed{\;\mathcal L(\theta) = -\frac1{N}\sum_{i} R_\theta(\tau^E_i) \;+\; \operatorname{logsumexp}_{j\in M\cup E}\big(R_\theta(\tau_j) + \log z_j\big) - \log(M{+}N)\;}
$$

### 2.3 Implementazione

- **Buffer accoppiati a finestra fissa** (`demo_algorithm.py:117-118`): `_model_buffer` (rollout per iterazione) e `_policy_window` (snapshot congelati della policy che li ha generati), entrambi `deque(maxlen=policy_window=10)`. Evizione in lock-step → la mistura copre esattamente le policy che hanno prodotto i campioni nel buffer.
- **Pesi di importance** (`_recompute_fusion_weights`, `:287-309`): calcola $\log z_j = \log W - \operatorname{logsumexp}_\kappa \sum_t\log\pi_\kappa$ per ogni traiettoria (buffer + esperti), una volta per iterazione (non dipendono da $\theta$).
- **Loss** (`_maxent_loss_on`, `:331-372`): primo termine $-\overline{R_\theta(\tau^E)}$; secondo termine `logsumexp(all_returns + all_log_iw) - log(n_total)` con `all_returns = cat([model_returns, expert_returns])`.
- **Centering numerico** (`:368`): `all_log_iw -= all_log_iw.mean()` prima del `logsumexp`.
- **Validazione deterministica** (`_evaluate_reward_model`, `:196`): snapshot fisso campionato una volta.

**Pseudo-algoritmo dell'implementazione:**

```
Algoritmo 2 — DemoAlgorithm (GCL / MaxEnt IRL)
push_data(self.trajectories):                # ogni iterazione, PRIMA di train_agent
  _model_buffer.append(rollout corrente)     # deque maxlen = policy_window (=10)
  _policy_window.append(_snapshot_policy(π)) # snapshot congelato (state_dict), lock-step
  _update_importance_weights():              # i pesi NON dipendono da θ → 1 volta/iter
    per ogni τ in (_model_buffer ∪ demo):
      log z_τ = log W - logsumexp_κ ( Σ_t log π_κ(a_t|s_t) )   # mistura ultime W policy

train_reward_model():
  se _model_buffer vuoto: return            # salta la prima iterazione
  per ogni membro m:
    ripeti gradient_steps_rew (=100) volte:
      model_R  = [ Σ_t r_θ(s,a)  per τ in _model_buffer ]      # forward grezzo
      expert_R = [ Σ_t r_θ(s,a)  per τ in demo ]
      all_R    = cat(model_R, expert_R)                        # demo incluse in Z (Alg.2)
      all_logz = cat(buffer_logz, demo_logz)
      all_logz -= mean(all_logz)                               # centering numerico
      log_Z = logsumexp(all_R + all_logz) - log(N+M)
      loss  = -mean(expert_R) + log_Z                          # -E[R^E] + log Z
      θ ← θ - lr_rew · clip(∇loss, grad_clip_rew=1.0)
reward per PPO: r_θ(s,a) normalizzato in EnvRewardWrapper
```

### 2.4 Mappatura teoria → codice

| Concetto teorico | Implementazione |
|---|---|
| $R_\theta(\tau)=\sum_t r_\theta$ | `_traj_sum_reward` (`:311-317`) |
| $-\frac1N\sum_i R_\theta(\tau^E_i)$ | `-expert_returns.mean()` (`:372`) |
| $\log Z \approx \operatorname{logsumexp}(R+\log z)-\log M$ | `log_z = th.logsumexp(all_returns+all_log_iw)-np.log(n_total)` (`:370`) |
| Demo aggiunte ai campioni (Alg.2 step 4) | `cat([model_returns, expert_returns])` (`:361`) |
| Fusione $q_{\text{mix}}$ delle ultime $W$ policy | `_policy_window` + `_recompute_fusion_weights` |
| $\log z_j = \log W - \operatorname{logsumexp}_\kappa \log q_\kappa$ | riga `:307` |
| Snapshot policy congelata | `_snapshot_policy` (`:256-262`) |

### 2.5 Verifica di correttezza e problemi

1. **Segno e struttura della loss: corretti.** Si minimizza $-\mathbb E_{\tau^E}[R_\theta] + \log\hat Z$. Il gradiente spinge il reward su per le demo e giù per i campioni della mistura — esattamente la stima MaxEnt. ✓

2. **Demo incluse nella stima di $Z$ (corretto e necessario).** Aggiungere il batch esperto all'insieme dei campioni nel `logsumexp` (`:360-361`) è prescritto da Algorithm 2 (step 4) di GCL e impedisce che l'obiettivo diventi illimitato (altrimenti basterebbe alzare $R_\theta$ all'infinito sulle demo). ✓ Documentato nel docstring `:338-339`.

3. **Centering dei pesi di importance (corretto, ma cambia il valore della loss).** Sottrarre `all_log_iw.mean()` dentro il `logsumexp` aggiunge una costante a $\log Z$ **indipendente da $\theta$** ⇒ il **gradiente è identico** e la stabilità numerica migliora. Tuttavia il **valore** della loss è traslato di quella costante. È irrilevante per l'ottimizzazione, ma rende `loss_val` non direttamente confrontabile fra batch con costanti diverse — il codice lo aggira usando uno **snapshot di validazione fisso** (`:196-219`). Coerente e ben motivato.

4. **Importance weighting via mistura di policy (approssimazione teorica — punto critico, CONFERMATO EMPIRICAMENTE).** Le demo ricevono $\log z_j$ calcolato **sotto la mistura di policy** (`:302` le accoda a `trajectories`), ma la vera distribuzione generatrice delle demo è quella dell'esperto, **non** la mistura. È un'approssimazione esplicitamente dichiarata (docstring `:24-25`, `:296`). In GCL le demo entrano nel sample set con la stessa background distribution; pesarle sotto $q_{\text{mix}}$ è una scelta pragmatica. **Impatto teorico:** se la policy diverge dall'esperto, $q_{\text{mix}}(\tau^E)$ è minuscola ⇒ $\log z_j$ delle demo molto grande ⇒ le demo dominano il `logsumexp` e l'estimatore di $Z$ diventa ad alta varianza.

   **Questo è esattamente ciò che accade nella pratica — è la causa primaria del collasso del training (vedi §2.6 per i dati).** Il problema è più grave del previsto: la degenerazione dei pesi di fusione **non riguarda solo le demo ma l'intero estimatore**, perché $\log q_\kappa(\tau)=\sum_t\log\pi_\kappa(a_t\mid s_t)$ è un **prodotto su traiettorie intere** in spazio d'azione continuo. Su orizzonti lunghi (episodi di media ~160 step) la somma dei log-prob varia di centinaia in scala-log tra traiettorie e tra policy ⇒ il $\operatorname{logsumexp}_\kappa$ è dominato da una singola policy e, a valle, il `logsumexp` di partizione da un **singolo campione**. L'**effective sample size** dei pesi è $\approx 0$ (`fusion/ess_frac` $\sim 10^{-3}$) **fin dalla prima iterazione**. L'estimatore MaxEnt-IRL a fusione con pesi su traiettoria intera è quindi **statisticamente inutilizzabile in questo dominio** (orizzonte lungo + azioni continue), indipendentemente dagli iperparametri.

5. **Coerenza buffer↔finestra policy (corretto).** I due deque evictano in lock-step (`:117-118`, append in `push_data` `:146` + `_update_importance_weights` `:147`), quindi la mistura copre **esattamente** le policy che hanno generato i campioni nel buffer. La chiamata avviene in `push_data`, *prima* di `train_agent`, quando la policy è ancora $\pi_\kappa$ che ha generato `self.trajectories` — corretto.

6. **Reward grezzo in `forward`, normalizzato solo in `predict`.** La loss usa `member(...)` = `forward` (grezzo), così il `logsumexp` lavora sui return reali. La normalizzazione (`NormalizedRewardNet`) interviene solo in inference/PPO. ✓ Coerente.

7. **`fragment_length: 100` non influisce sulla loss IRL.** In Demo, `collect_feedback` ritorna `([],[])` (`:133`): il fragmenter **non** è usato per la loss (che opera su traiettorie intere). `fragment_length` impatta solo il logging di correlazione del reward. Possibile fonte di confusione, non un bug.

8. **`grad_clip_rew: 1.0`** attivo (config) — clipping della norma del gradiente del reward (`:165-166`), stabilizza ma non altera l'obiettivo.

**Verdetto:** l'implementazione è **fedele e matematicamente corretta** rispetto a GCL/MaxEnt IRL (segni, loss, demo-in-$Z$, fusione, centering). Tuttavia, l'**estimatore di $Z$ a fusione con importance sampling su traiettorie intere non funziona in questo dominio** (orizzonte lungo + azioni continue): la verifica empirica (§2.6) mostra che la degenerazione dei pesi (ESS$\approx0$) è la **causa diretta del collasso del training**. Non è un bug di codice ma un **limite metodologico** dell'algoritmo applicato a questo task. L'ablazione a pesi uniformi (§2.6) lo conferma e fornisce la via di riparazione.

### 2.6 Evidenza empirica: diagnosi del collasso e ablazione

Questa sezione documenta la verifica sperimentale del punto §2.5.4, con strumentazione aggiunta apposta al codice.

**Strumentazione diagnostica aggiunta** (`demo_algorithm.py`): decomposizione della loss e salute dell'estimatore, loggate ogni iterazione —
- `reward/expert_return`, `reward/model_return`, `reward/return_gap` (= expert − model): il **vero segnale IRL** (deve essere $>0$);
- `reward/log_z`: termine di partizione (deve restare limitato);
- `fusion/ess_frac`, `fusion/log_iw_std`: effective sample size e dispersione dei pesi di fusione.

*(Nota implementativa collaterale: `_snapshot_policy` è stata convertita da `copy.deepcopy` del modulo a snapshot dello `state_dict`, perché le policy SB3 contengono tensori non-leaf nella `action_dist` cache che `deepcopy` non può copiare — bug che faceva abortire il run all'iterazione 10.)*

**Run di riferimento `PPO_demo/i63qw9v2` (pesi di fusione attivi):**

| iter | timesteps | success | `ess_frac` | `log_iw_std` | `return_gap` | `log_z` | `kendall` |
|---|---|---|---|---|---|---|---|
| 0 | 20k | 0.00 | **0.000** | 14 | **−44** | 402 | −0.31 |
| 4 | 100k | 0.21 | **0.000** | 63 | **−117** | 240 | −0.07 |
| 8 | 180k | **0.35** | **0.000** | 99 | **−147** | 205 | 0.09 |
| 9 | 200k | **0.03** | 0.000 | 102 | −145 | 210 | 0.08 |
| 12 | 260k | 0.07 | 0.001 | 181 | −137 | **1207** | 0.05 |
| 18 | 380k | 0.00 | 0.001 | 142 | −140 | **1482** | 0.11 |

Lettura come catena causale: `ess_frac`$\approx0$ **da subito** ⇒ la partizione è dominata da un singolo campione (spesso una demo) ⇒ minimizzare quel termine **abbassa il reward dell'esperto** ⇒ `return_gap` negativo (reward col **segno invertito**: l'agente è premiato più dell'esperto) ⇒ `kendall`$\approx0$ (il reward non impara nulla di utile). Quando `log_iw_std` supera ~165 (iter 11–12) la partizione **esplode** (`log_z`: 210→1207) e con essa la loss, distruggendo definitivamente la policy. Il picco di success a 0.35 (iter 8) **non** è merito del reward appreso: è PPO + lo shaping `fast` dell'ambiente.

**Ablazione a pesi uniformi (`importance_weighting: false`).** È stato aggiunto un flag che forza $\log z_j=0$ per tutte le traiettorie (la partizione diventa una media semplice sul buffer), saltando l'estimatore di fusione. Run `PPO_demo/5cajtref`:

| metrica | fusione attiva (`i63qw9v2`) | pesi uniformi (`5cajtref`) |
|---|---|---|
| `return_gap` (expert − agent) | **−44 → −150** (segno invertito) | **+1.8 → +3.2** (corretto) |
| `log_z` | esplode 210 → **1500** | limitato **~1–2.5** |
| `kendall` (reward vs vero) | **≈0** sempre | sale a **~0.70** |
| picco di success | 0.35, poi collasso a ~0.1 | **0.81** (a ~540k) |

Spegnere l'estimatore di fusione **inverte ogni diagnostica**: il reward impara un ordinamento corretto (Kendall 0.70), la partizione resta limitata, il successo più che raddoppia. **Conferma che i pesi di fusione su traiettoria intera erano la causa unica del collasso primario**, e che il confound lunghezza/somma-return (ipotesi alternativa) **non** è il blocco.

**Collasso secondario (distinto, lato policy).** Anche con pesi uniformi il run non è pulito fino in fondo: dopo il picco di 0.81 (~540k) il successo **ricollassa** verso ~0.03 (~860k), ma con `return_gap` ancora positivo e `kendall`~0.70 — quindi **il reward è ancora buono, è la policy a cedere**. Cause: (i) la ricompensa appresa ha **range dinamico troppo piccolo** sulla distribuzione on-policy (`mean_model_reward` ~±1 contro reward vero ~±35), aggravato dalla ri-centratura della media a ogni iterazione (`before_agent_training`), e (ii) **collasso dell'entropia** (la policy `std` scende 0.44→0.34 *prima* del crollo del successo). È la non-stazionarietà classica dell'IRL: quando l'allievo raggiunge il maestro, il segnale discriminante si appiattisce on-distribution e ogni deriva della policy resta non corretta. Leve da esplorare: rimuovere la ri-centratura per-iterazione, aumentare la scala/range del reward, alzare `ent_coef`, oppure early-stopping al picco.

**Conclusione per la difesa.** Il MaxEnt-IRL a fusione è teoricamente corretto ma **non scala a orizzonti lunghi in azione continua**: l'IS su traiettoria intera è degenere (ESS$\approx0$). Le vie di riparazione, in ordine di principio: (a) abbandonare i pesi su traiettoria intera (uniformi, oppure per-step / a frammenti di lunghezza fissa, oppure self-normalized con clip dei pesi); (b) usare un estimatore di $Z$ **non basato su IS** — ed è precisamente ciò che fanno GAIL/AIRL (§3–4) sostituendo la partizione con un **discriminatore**, motivazione che giustifica il passaggio a quei metodi.

---

## 3. GailAlgorithm — Generative Adversarial Imitation Learning (Ho & Ermon 2016)

**File:** `algorithms/gail_algorithm.py` · **Paper:** `Gail.pdf`.

### 3.1 Obiettivo

Invece di stimare un reward esplicito, si fa **matching della distribuzione di occupazione** stato-azione tra policy ed esperto, tramite un gioco avversariale tra un discriminatore $D$ e la policy $\pi$.

### 3.2 Formulazione matematica

**Obiettivo GAIL (Ho & Ermon, Eq. 18).** Con regolarizzazione di entrofia $H(\pi)$:

$$
\min_{\pi}\max_{D\in(0,1)} \;\; \mathbb E_{\pi}\big[\log D(s,a)\big] + \mathbb E_{\pi_E}\big[\log(1-D(s,a))\big] - \lambda H(\pi).
$$

Il discriminatore è addestrato a **classificare** le transizioni: $D(s,a)\to 1$ se vengono dalla policy, $\to 0$ se vengono dall'esperto. Per $D$ fissato, la policy minimizza $\mathbb E_\pi[\log D(s,a)]$, equivalente a **massimizzare il reward surrogato**:

$$
r(s,a) = -\log D(s,a).
$$

All'ottimo $D^*(s,a) = \dfrac{\rho_\pi(s,a)}{\rho_\pi(s,a)+\rho_{\pi_E}(s,a)}$; il punto di sella si ha quando $\rho_\pi=\rho_{\pi_E}$ (occupancy matching).

**Loss del discriminatore (binary cross-entropy):**

$$
\mathcal L_D = -\,\mathbb E_{\pi}\big[\log D(s,a)\big] - \mathbb E_{\pi_E}\big[\log(1-D(s,a))\big],
$$

con $D(s,a)=\sigma(\text{logit}_\phi(s,a))$.

Il **logit** è l'uscita scalare di una rete MLP parametrizzata da $\phi$, che prende in input la concatenazione di stato e azione:

$$
\text{logit}_\phi(s,a) = f_\phi\big([\,s \,;\, a\,]\big) \in \mathbb R,
$$

dove $[\,s\,;\,a\,]\in\mathbb R^{\dim(s)+\dim(a)}$ è il vettore ottenuto concatenando $s$ e $a$, e $f_\phi$ è una MLP (di default `[128,128]`, attivazione `tanh`) con un singolo neurone di uscita lineare (nessuna sigmoide nella rete). La sigmoide $\sigma(\cdot)$ è applicata implicitamente dalla loss `binary_cross_entropy_with_logits`, che lavora direttamente sul logit grezzo per stabilità numerica; analogamente il reward $-\log D(s,a)=\text{softplus}(-\text{logit}_\phi)$ è calcolato dal logit senza materializzare $D$. Implementazione: `StateActionDiscriminatorNet.forward` (`reward_nets.py:150-152`).

### 3.3 Implementazione

- **Discriminatore:** `make_gail_discriminator_ensemble` → ensemble di `StateActionDiscriminatorNet` (input **solo** $(s,a)$, `reward_nets.py:134-152`), avvolto in `GailRewardNet`.
- **Loss** (`gail_algorithm.py:49-75`): convenzione di etichetta del **paper originale** — esperto → label 0, agente → label 1.

```python
logits_e = member(obs_e, act_e, ns_e, done_e)
loss_e = F.binary_cross_entropy_with_logits(logits_e, th.zeros_like(logits_e))  # expert → 0
logits_a = member(obs_a, act_a, ns_a, done_a)
loss_a = F.binary_cross_entropy_with_logits(logits_a, th.ones_like(logits_a))   # agent  → 1
return loss_e + loss_a
```

- **Reward per PPO** (`GailRewardNet.predict`, `reward_nets.py:309-319`): `softplus(-logits)` $= \log(1+e^{-\text{logit}}) = -\log\sigma(\text{logit}) = -\log D(s,a)$. ✓ esattamente il reward GAIL.
- **No importance sampling:** `_update_importance_weights` → `pass` (`:39-42`).
- **Campioni agente:** ricampionati a ogni gradient step dalla **rollout corrente** (`_batch_transitions` da `self.trajectories`), on-policy.
- **No normalizzazione del reward model:** `before_agent_training` → `pass` (`:81-82`); la normalizzazione avviene solo in `EnvRewardWrapper`.

**Pseudo-algoritmo dell'implementazione:**

```
Algoritmo 3 — GailAlgorithm
push_data: eredita da DemoAlgorithm (riempie _model_buffer, ma è INUTILIZZATO)
_update_importance_weights(): pass        # niente importance sampling

train_reward_model():                     # addestra il discriminatore D_φ
  per ogni membro D_φ:
    ripeti gradient_steps_rew (=100) volte:
      (s,a)_E ← batch da demo
      (s,a)_A ← _batch_transitions(self.trajectories)     # on-policy, rollout corrente
      logit_E = D_φ(s,a)_E ;  logit_A = D_φ(s,a)_A
      loss_E = BCE_with_logits(logit_E, 0)                 # esperto → 0  (paper Ho&Ermon)
      loss_A = BCE_with_logits(logit_A, 1)                 # agente  → 1
      φ ← φ - lr_rew · ∇(loss_E + loss_A)
before_agent_training(): pass
reward per PPO: r(s,a) = softplus(-logit_φ(s,a)) = -log D(s,a)   # poi normalizzato
```

### 3.4 Mappatura teoria → codice

| Concetto teorico | Implementazione |
|---|---|
| $D(s,a)=\sigma(\text{logit})$, prob. "policy" | `StateActionDiscriminatorNet` (logit) + $\sigma$ implicita in BCE |
| esperto → 0, policy → 1 | `zeros_like(logits_e)`, `ones_like(logits_a)` ✓ paper |
| $\mathcal L_D$ (BCE) | `binary_cross_entropy_with_logits` (`:61,71`) |
| reward $-\log D(s,a)$ | `softplus(-logits)` (`:319`) ✓ |
| input $(s,a)$ soltanto | `th.cat([state, action])` (`:151`) ✓ |
| nessun IS / nessuna mistura | `_update_importance_weights` no-op |

### 3.5 Verifica di correttezza e problemi

1. **Convenzione delle etichette: corretta e coerente col paper.** Il docstring del file riporta letteralmente $\max_D \mathbb E_{\text{agent}}[\log D] + \mathbb E_{\text{expert}}[\log(1-D)]$ (agente → $D$ alto). Codice: agente label 1, esperto label 0 ⇒ il discriminatore è addestrato a far $D(\text{agent})\to1$, $D(\text{expert})\to0$. Reward $-\log D$: l'agente massimizza il reward abbassando $D(s,a)$, cioè facendosi scambiare per esperto. **Tutto consistente.** ✓ (Nota: la libreria `imitation` usa la convenzione opposta esperto=1; qui si segue l'originale Ho & Ermon — vedi §8 Q.)

2. **Reward sempre positivo ⇒ "survival bias" (punto critico noto).** $r=-\log D = \text{softplus}(-\text{logit}) > 0$ sempre. Con reward strettamente positivo, PPO è incentivato a **prolungare gli episodi** indipendentemente dall'imitazione. È esattamente la critica che AIRL muove a GAIL. In un task di guida (dove sopravvivere = non collidere/uscire di strada) può essere benigno, ma è un bias da dichiarare. `ent_coef: 0.01` (config) reintroduce un minimo di $H(\pi)$ del termine $-\lambda H(\pi)$.

3. **Discriminatore su $(s,a)$ senza `next_status`/`done`.** Coerente col GAIL originale (occupancy su stato-azione). Da notare: lo `SumoRewardNet` degli altri algoritmi usa anche `next_status` e `done`; qui no. Scelta corretta per GAIL.

4. **Buffer MaxEnt ereditato ma inutilizzato (implementativo).** GAIL eredita `DemoAlgorithm.push_data`, che continua a riempire `_model_buffer` e a chiamare `_update_importance_weights` (no-op). Il guard `if not self._model_buffer: return` in `train_reward_model` serve solo a saltare la prima iterazione. La loss GAIL **ignora** il buffer e ricampiona da `self.trajectories`. Nessun bug, ma memoria sprecata e codice fuorviante (il buffer cresce senza scopo).

5. **`min(n, len)` con `replace` (minore).** `_batch_transitions` (`:88-94`): `n = min(n, len(all_t))` e poi `replace = len(all_t) < n` è **sempre `False`** dopo il `min`. Innocuo, ma la logica di replacement è morta.

6. **Discriminatore vs policy: aggiornamento alternato.** Il discriminatore fa `gradient_steps_rew=100` passi per iterazione, poi PPO fa `timesteps_per_iteration` passi. Squilibrio D/π elevato: un discriminatore troppo forte può saturare ($D\to\{0,1\}$, gradienti $\to0$, reward $-\log D$ esplosivo/nullo). Iperparametro da monitorare (instabilità avversariale classica).

**Verdetto:** implementazione **corretta e aderente al GAIL originale** (etichette, BCE, reward $-\log D$, occupancy su $(s,a)$). Le criticità sono quelle *intrinseche* di GAIL (survival bias, instabilità avversariale), non errori del codice. Il buffer IRL ereditato e inutilizzato è un debito tecnico.

---

## 4. AirlAlgorithm — Adversarial Inverse RL (Fu et al. 2018)

**File:** `algorithms/airl_algorithm.py` · reward net in `reward_nets.py::AirlRewardNet` · **Paper:** `Airl.pdf`.

### 4.1 Obiettivo

Come GAIL, ma con un discriminatore **strutturato** che recupera un reward esplicito e robusto. Risolve due problemi di GAIL: (i) il reward non è recuperabile, (ii) il survival bias.

### 4.2 Formulazione matematica

**Discriminatore AIRL (Fu et al., Eq. 4).** Forma vincolata:

$$
D_\theta(s,a,s') = \frac{\exp\big(f_\theta(s,a,s')\big)}{\exp\big(f_\theta(s,a,s')\big) + \pi(a\mid s)},
$$

dove $f$ ha la struttura **reward + shaping** (Eq. 14):

$$
f_{\theta,\phi}(s,a,s') = g_\theta(s,a) + \gamma\, h_\phi(s') - h_\phi(s),
$$

con $g_\theta$ = reward, $h_\phi$ = potenziale di shaping (una "value function"). Si nota che

$$
\text{logit}\,D = f_\theta(s,a,s') - \log\pi(a\mid s).
$$

**Loss (cross-entropy del discriminatore, esperto → 1, policy → 0):**

$$
\mathcal L_D = -\,\mathbb E_{\pi_E}\big[\log D_\theta\big] - \mathbb E_{\pi}\big[\log(1-D_\theta)\big].
$$

**Reward per la policy** (entropy-regularized, Eq. 6):

$$
\hat r(s,a,s') = \log D_\theta - \log(1-D_\theta) = f_\theta(s,a,s') - \log\pi(a\mid s).
$$

**Risultato di disentanglement (Teorema 5.1):** se $g_\theta=g(s)$ dipende **solo dallo stato**, all'ottimo $g$ recupera il reward vero a meno di costante e $h$ recupera la value function, rendendo il reward trasferibile a nuove dinamiche.

### 4.3 Implementazione

- **Reward net** `AirlRewardNet` (`reward_nets.py:388-429`):
  - `reward_net` = $g(s,a)$ (input $(s,a)$, riga `:406-408`),
  - `value_net` = $h(s)$ (input $s$, riga `:401`),
  - `shaped_reward = r + γ(1-done)·V(s') - V(s)` $= f$ (`:410-414`),
  - `discriminator_logit = shaped_reward - policy_log_prob` $= f - \log\pi$ (`:425-426`),
  - `forward = discriminator_logit - mean` (centratura, `:428-429`).
- **Loss** (`airl_algorithm.py:87-106`): esperto → 1, policy → 0.

```python
f_e = member.shaped_reward(obs_e, act_e, next_obs_e, done_e)
f_a = member.shaped_reward(obs_a, act_a, next_obs_a, done_a)
logits_e = f_e - self._policy_log_prob(obs_e, act_e)   # f - log π
logits_a = f_a - self._policy_log_prob(obs_a, act_a)
loss_e = F.binary_cross_entropy_with_logits(logits_e, th.ones_like(logits_e))   # expert → 1
loss_a = F.binary_cross_entropy_with_logits(logits_a, th.zeros_like(logits_a))  # policy → 0
```

- **Reward per PPO:** `reward_model.predict` = `forward` = `discriminator_logit - mean` $= (f-\log\pi)-\text{mean}$. Coincide con $\hat r = \log D - \log(1-D)$ del paper (centrato). ✓
- **`next_state` reale:** `EnvRewardWrapper` rileva `uses_next_state=True` e passa l'osservazione successiva $s'$ (`env_wrappers.py:85-86`).
- **Centratura logit:** `before_agent_training` (`:68-81`) setta `_mean` = media del logit sul rollout corrente (riduce bias, mitiga il survival bias).
- **BC warmup opzionale** (`:36-62`): NLL sulle azioni esperte (config: `bc_warmup_steps: 0`, disattivo).

**Pseudo-algoritmo dell'implementazione:**

```
Algoritmo 4 — AirlAlgorithm
(opzionale) BC warmup: bc_warmup_steps passi di NLL sulle azioni esperte (disattivo)
_update_importance_weights(): pass        # niente importance sampling

train_reward_model():                     # discriminatore strutturato (g + shaping)
  per ogni membro:
    ripeti gradient_steps_rew (=100) volte:
      (s,a,s',done)_E ← batch demo ;  (s,a,s',done)_A ← _batch_airl_transitions(traj)
      f = g_θ(s,a) + γ·(1-done)·h_φ(s') - h_φ(s)          # reward + potential shaping
      logit = f - log π(a|s)                              # log π DETACHED
      loss_E = BCE_with_logits(logit_E, 1)                # esperto → 1
      loss_A = BCE_with_logits(logit_A, 0)                # agente  → 0
      (θ,φ) ← update con lr_rew
before_agent_training():
  _mean ← media del logit (f - log π) sul rollout corrente   # centratura → mitiga survival bias
reward per PPO: r̂(s,a,s') = (f - log π) - _mean = log D - log(1-D)   # poi normalizzato
```

### 4.4 Mappatura teoria → codice

| Concetto teorico | Implementazione |
|---|---|
| $g_\theta(s,a)$ (reward) | `AirlRewardNet.reward` → `reward_net` |
| $h_\phi(s)$ (shaping/value) | `AirlRewardNet.value_net` |
| $f = g + \gamma(1{-}d)h(s') - h(s)$ | `shaped_reward` (`:410-414`) |
| $\text{logit}D = f - \log\pi$ | `discriminator_logit` (`:425-426`) |
| esperto → 1, policy → 0 | `ones_like(logits_e)`, `zeros_like(logits_a)` ✓ |
| $\hat r = \log D - \log(1-D) = f - \log\pi$ | `forward`/`predict` (`:428`) |
| $s'$ reale necessario | `uses_next_state = True` (`:389`) |
| $\log\pi$ detached | `_policy_log_prob` con `.detach()` (`:108-111`, `reward_nets.py:423`) |

### 4.5 Verifica di correttezza e problemi

1. **Struttura del discriminatore: fedele al paper.** $f = g + \gamma(1-d)V(s') - V(s)$, $\text{logit}=f-\log\pi$, BCE con esperto=1/policy=0, reward $=f-\log\pi$. Tutti i segni e le forme combaciano con Fu et al. ✓ Il fattore $(1-\text{done})$ azzera $V(s')$ a fine episodio (`:414`), corretto; per le transizioni terminali `_batch_airl_transitions` mette $s'=0$ come placeholder sicuro (`:129-131`).

2. **Reward state-action, NON state-only ⇒ niente disentanglement (punto critico).** Il Teorema 5.1 di AIRL garantisce reward **trasferibile** solo se $g_\theta = g(s)$ dipende **solo dallo stato**. Qui `reward_net` prende $(s,a)$ (`:406-407`), quindi si ottiene la variante "non disentangled". **Impatto:** ottima per imitazione nello stesso MDP, ma il reward recuperato **non è garantito trasferibile** a dinamiche diverse — proprio la feature di punta di AIRL. Da segnalare esplicitamente al relatore: è una scelta che indebolisce la motivazione "AIRL vs GAIL".

3. **$\log\pi$ detached: corretto.** Il gradiente del discriminatore non deve fluire nella policy (`.detach()`, `:111`); la policy è aggiornata separatamente da PPO. ✓

4. **Coerenza di $\pi$ usata nel logit.** In training si usa `trajectory_generator.agent.policy` (la policy corrente, `:109`); in inference `AirlRewardNet._policy_log_prob` usa `self.policy` settata da `EnvRewardWrapper.set_policy` (`env_wrappers.py:68-69`). Entrambe puntano alla stessa policy PPO. ✓ Coerente. (Sottigliezza: la policy si aggiorna durante PPO, quindi il termine $-\log\pi$ del reward cambia lungo l'iterazione; è il comportamento atteso in AIRL.)

5. **Doppia centratura del reward.** `AirlRewardNet.forward` sottrae `_mean`, e poi `EnvRewardWrapper` ri-normalizza a media 0/std 1. Ridondante ma innocuo (entrambe sono shift/scale).

6. **`gamma` duplicato.** `gamma: 0.997` è specificato sia in `reward_model_kwargs` (per lo shaping AIRL) sia in `agent.kwargs` (per PPO/GAE). Devono coincidere (e coincidono in config); se divergessero, lo shaping AIRL userebbe un $\gamma$ diverso dalla value function di PPO — incoerenza da evitare. Attenzione in eventuali ablation.

7. **Buffer MaxEnt ereditato inutilizzato:** identico a GAIL (§3.5 p.4) — AIRL ricampiona da `self.trajectories` (`_sample_agent_batch`) e ignora `_model_buffer`.

8. **`replace=False` in `_batch_airl_transitions` (`:141`).** Campiona senza rimpiazzo: se `n > #transizioni` darebbe errore, ma `n=min(n,len)` lo previene (`:140`). OK.

**Verdetto:** implementazione del **discriminatore AIRL corretta** in ogni segno e nella struttura reward+shaping; il reward per PPO è esattamente $f-\log\pi$. L'unico scostamento sostanziale dal paper è l'uso di $g(s,a)$ invece di $g(s)$, che **rinuncia alla proprietà di disentanglement/trasferibilità** — il limite teorico più importante da saper difendere.

---

## 5. DaggerAlgorithm — Dataset Aggregation (Ross et al. 2011)

**File:** `algorithms/dagger_algorithm.py` · **Paper:** `Ross et al. - 2011` (*A Reduction of Imitation Learning ... to No-Regret Online Learning*).

> Non ha config `test_*` dedicato; è invocato da `scripts/eval.py`. Non eredita da `BaseRewardLearningAlgorithm` (è imitation supervisionata pura, **nessun reward appreso, nessun PPO**).

### 5.1 Obiettivo

La behaviour cloning ingenua soffre di **distribution shift**: addestrata sugli stati dell'esperto, la policy commette errori che la portano in stati mai visti, dove sbaglia ancora di più (errore composto $O(T^2\epsilon)$). DAgger risolve raccogliendo dati **sugli stati visitati dalla policy stessa**, etichettati dall'esperto, e aggregandoli (riduzione a online learning no-regret, errore $O(T\epsilon)$).

### 5.2 Formulazione matematica

A ogni round $i$ si usa una policy mista

$$
\pi_i = \beta_i\,\pi^* + (1-\beta_i)\,\hat\pi_i,
$$

con $\pi^*$ esperto, $\hat\pi_i$ policy appresa, $\beta_i$ decrescente ($\beta_0=1$ ⇒ primo round pura esperta). Si esegue $\pi_i$ nell'ambiente, ma **si registra l'azione dell'esperto** $\pi^*(s)$ su ogni stato visitato:

$$
\mathcal D \leftarrow \mathcal D \cup \{(s, \pi^*(s)) : s\sim d_{\pi_i}\}.
$$

Si riaddestra con **behaviour cloning** (NLL) sull'**intero** dataset aggregato:

$$
\hat\pi_{i+1} = \arg\max_{\pi}\; \mathbb E_{(s,a^*)\sim\mathcal D}\big[\log\pi(a^*\mid s)\big].
$$

### 5.3 Implementazione

- **Schedule di $\beta$** (`:147-149`): `beta = beta_decay ** round_idx`, decadimento esponenziale ($\beta_0=1$, `beta_decay=0.7`).
- **Raccolta mista** (`_collect_trajectories`, `:151-215`): a ogni step si calcola sia `agent_action` sia `expert_action`; si **esegue** l'azione esperta con prob. $\beta$, altrimenti quella dell'agente (`:193`); ma nella transizione si salva **sempre l'azione dell'esperto** (`expert_action_for_transition`, `:204`). ✓ È esattamente il meccanismo DAgger.
- **Aggregazione** (`:97-99`): `self.dataset.extend(traj.transitions)` — il dataset **cresce monotonamente** (mai svuotato). ✓
- **BC** (`_bc_train`, `:217-264`): `loss = -log_prob.mean()` su minibatch dell'intero dataset, per `bc_epochs` epoche. ✓
- **Valutazione deterministica** (`_evaluate`, `:266-296`).

**Pseudo-algoritmo dell'implementazione:**

```
Algoritmo 5 — DaggerAlgorithm (no reward, no PPO)
D ← ∅                                       # dataset aggregato, cresce monotonamente
per ogni round i = 0 … n_rounds-1:
  β = beta_decay ** i                        # β0 = 1 (primo round = pura esperta)
  _collect_trajectories con π_i:
    a ogni step su stato s:
      a_agent  = π̂(s)
      a_expert = π*(s)                        # esperto interrogabile online
      esegui a_expert con prob. β, altrimenti a_agent     # π_i = β·π* + (1-β)·π̂
      salva la transizione con etichetta = a_expert SEMPRE
  D ← D ∪ transizioni                         # aggregazione
  _bc_train(): ripeti bc_epochs su minibatch di tutto D:
    loss = -mean( log π̂(a*|s) )               # behaviour cloning (NLL)
    aggiorna π̂                                # niente reward, niente PPO
```

### 5.4 Mappatura teoria → codice

| Concetto teorico | Implementazione |
|---|---|
| $\pi_i = \beta\pi^* + (1-\beta)\hat\pi$ | `executed_action = expert if rng<beta else agent` (`:193`) |
| etichetta esperta $\pi^*(s)$ su stati di $\pi_i$ | `expert_action_for_transition` salvata sempre (`:204`) |
| $\beta_i$ decrescente | `beta_decay ** round_idx` (`:149`) |
| aggregazione $\mathcal D \cup$ | `self.dataset.extend(...)` (`:98`) |
| BC: $\max \log\pi(a^*\mid s)$ | `loss = -log_prob.mean()` (`:242`) |

### 5.5 Verifica di correttezza e problemi

1. **Meccanismo DAgger corretto.** Esecuzione mista + etichettatura esperta su stati on-policy + aggregazione monotona + BC sull'intero dataset: è la formulazione canonica di Ross et al. ✓

2. **Disagreement misurato ma non usato.** `disagreement_rate` è solo loggato (`:190`), non guida l'apprendimento. Metrica diagnostica. OK.

3. **`clip_grad_norm_(..., max_norm=inf)` (implementativo).** Riga `:246-248`: clipping con norma infinita **non clippa nulla**, serve solo a *misurare* `grad_norm`. Comportamento voluto ma non ovvio; non è un bug.

4. **Costo di riaddestramento crescente.** Si riaddestra da capo per `bc_epochs` sull'intero dataset aggregato a ogni round ⇒ costo per round cresce linearmente col round. Coerente con DAgger ("batch" version), ma scalabilità limitata.

5. **Nessun reward / nessun PPO.** DAgger è l'unico metodo che **non** impara un reward: presuppone un esperto **interrogabile online** ($\pi^*(s)$ su qualsiasi stato), assunzione molto più forte degli altri (che richiedono solo dimostrazioni offline o giudizi di preferenza). Questo è il principale **trade-off** da evidenziare nel confronto.

6. **Dipendenza opzionale da `sumo_rl_ego`.** L'import dell'esperto rule-based è in `try/except` (`:12-17`); se assente, `_expert_is_rule_based=False` e si usa il ramo vettoriale. Robusto.

**Verdetto:** implementazione **corretta e canonica** di DAgger. Il limite non è implementativo ma di *setting*: richiede un esperto interrogabile, ipotesi non sempre realistica.

---

## 6. Confronto tra algoritmi

| | **Preference** (Christiano) | **Demo** (GCL/MaxEnt) | **GAIL** (Ho&Ermon) | **AIRL** (Fu) | **DAgger** (Ross) |
|---|---|---|---|---|---|
| **Input richiesto** | giudizi di preferenza su coppie | traiettorie esperte offline | traiettorie esperte offline | traiettorie esperte offline | esperto **interrogabile online** |
| **Usa preferenze** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Usa dimostrazioni** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Reward esplicito appreso** | ✅ $r_\theta(s,a)$ | ✅ $r_\theta(s,a)$ | ⚠️ implicito $-\log D$ | ✅ $g(s,a)+$shaping | ❌ |
| **Discriminatore** | ❌ | ❌ | ✅ $D(s,a)$ | ✅ $D(s,a,s')$ strutturato | ❌ |
| **Modello statistico** | Bradley-Terry | MaxEnt $p\propto e^{R}$ | occupancy matching | MaxEnt via discriminatore | online no-regret |
| **Loss** | cross-entropy BT | $-\overline{R(\tau^E)}+\log Z$ | BCE discriminatore | BCE discriminatore (logit $=f-\log\pi$) | NLL (BC) |
| **Reward per PPO** | $r_\theta$ norm. | $r_\theta$ norm. | $-\log D$ | $f-\log\pi$ | — (no RL) |
| **Importance sampling** | bootstrap ensemble | ✅ fusione policy | ❌ | ❌ | ❌ |
| **RL outer** | PPO | PPO | PPO | PPO | nessuno (BC) |
| **Vantaggi** | non serve esperto agente; feedback economico | reward denso e recuperabile | matching robusto di occupancy | reward **potenzialmente** trasferibile; no survival bias | risolve distribution shift, $O(T\epsilon)$ |
| **Svantaggi** | molte query; ipotesi BT | stima di $Z$ ad alta varianza | reward non recuperabile; **survival bias**; instabile | implementato senza disentanglement (g dipende da a); avversariale | richiede esperto online; costo riaddestramento crescente |

**Asse concettuale unificante.** Christiano, Demo e AIRL condividono l'ipotesi $p(\tau)\propto e^{R_\theta(\tau)}$. Christiano la usa sul *rapporto* (BT), Demo sulla *densità* (con $Z$ esplicito), AIRL la ottiene come *punto fisso del discriminatore*. GAIL rinuncia al reward esplicito e fa solo matching di distribuzione. DAgger sta su un piano diverso (supervisionato, no reward).

---

## 7. Sintesi dei problemi trovati (per impatto)

| # | Algoritmo | File:funzione | Problema | Tipo | Impatto |
|---|---|---|---|---|---|
| 1 | AIRL | `reward_nets.py:406` `AirlRewardNet.reward` | $g$ dipende da $(s,a)$, non solo $s$ | **Concettuale** | Perde il disentanglement/trasferibilità (Teorema 5.1). Imitazione OK. |
| 2 | Demo | `demo_algorithm.py:302` `_recompute_fusion_weights` | demo pesate sotto la mistura di policy | **Concettuale** | Estimatore di $Z$ ad alta varianza se $\pi$ lontana dall'esperto. Approssimazione dichiarata. |
| 3 | GAIL | `gail_algorithm.py` (reward $-\log D$) | reward sempre positivo | **Concettuale** | Survival bias (critica AIRL). Mitigato da `ent_coef` e dal contesto. |
| 4 | Preference | `preference_algorithm.py:133` `fragment_avg_reward` | media invece di somma | **Matematico** | Nullo con `fragment_length=1`; rilevante solo per frammenti di lunghezza variabile. |
| 5 | GAIL/AIRL | `demo_algorithm.py:146` `push_data` | `_model_buffer` riempito ma inutilizzato | **Pratico** | Spreco di memoria + codice fuorviante. Nessun effetto sui risultati. |
| 6 | Condiviso | `env_wrappers.py:87` `_RunningMeanStd` | normalizzazione reward cumulativa, mai resettata | **Pratico** | Non-stazionarietà: la norm. si adatta lentamente al reward che cambia. |
| 7 | Preference | `preference_algorithm.py:150` ThreadPool | `max_workers=1` ≠ "parallel" | **Pratico** | Solo performance/commento. |
| 8 | GAIL | `gail_algorithm.py:92` `_batch_transitions` | logica `replace` morta dopo `min()` | **Pratico** | Innocuo. |
| 9 | AIRL | config | `gamma` duplicato (reward vs PPO) | **Pratico** | Rischio di incoerenza in ablation; ora coincidono. |
| 10 | DAgger | `dagger_algorithm.py:246` | `clip_grad_norm_(inf)` non clippa | **Pratico** | Voluto (solo misura); non ovvio. |

Nessun **errore di segno** o di massimizzazione/minimizzazione è stato riscontrato: tutte e cinque le loss hanno segno e direzione di ottimizzazione corretti rispetto ai rispettivi paper.

---

## 8. Domande probabili (con risposte)

### 8.1 Domande teoriche

**Q. Qual è l'ipotesi statistica comune a Christiano, MaxEnt IRL e AIRL?**
La razionalità di Boltzmann: $p(\tau)\propto\exp(R_\theta(\tau))$. Christiano la applica al confronto pairwise (Bradley-Terry), MaxEnt alla densità assoluta (introducendo $Z$), AIRL la ottiene come discriminatore ottimo.

**Q. Perché le preferenze e non le dimostrazioni?**
Le preferenze richiedono solo che l'oracolo *giudichi*, non che *sappia agire* in modo ottimo; sono adatte quando l'esperto è subottimo o quando dimostrare è costoso. Le dimostrazioni danno più informazione per campione ma presuppongono un esperto competente.

**Q. Cosa garantisce DAgger che la BC ingenua non garantisce?**
Riduce l'errore composto da $O(T^2\epsilon)$ a $O(T\epsilon)$ raccogliendo dati sulla distribuzione di stati indotta dalla policy stessa (no-regret online learning), eliminando il distribution shift.

**Q. Qual è la differenza chiave tra GAIL e AIRL?**
GAIL fa occupancy matching senza recuperare un reward (reward $-\log D$ implicito, soggetto a survival bias). AIRL struttura il discriminatore come $f=g+\gamma h(s')-h(s)$ e, se $g=g(s)$, recupera un reward disentangled e trasferibile.

### 8.2 Domande matematiche

**Q. Scriva la loss di Christiano e spieghi il segno.**
$\mathcal L=-\sum \mu_1\log\sigma(R^1-R^2)+\mu_2\log\sigma(R^2-R^1)$. È la NLL del modello Bradley-Terry; si minimizza ⇒ massima verosimiglianza delle preferenze.

**Q. Da dove viene il `logsumexp` in MaxEnt IRL?**
Da $\log Z=\log\int e^{R_\theta(\tau)}d\tau$, stimato per importance sampling: $\log Z\approx\operatorname{logsumexp}_j(R_\theta(\tau_j)-\log q(\tau_j))-\log M$. Il `logsumexp` è il log della media degli esponenziali dei return pesati.

**Q. Perché si possono centrare i pesi di importance prima del `logsumexp`?**
Perché $\log z_j$ non dipende da $\theta$: sottrarne la media aggiunge una costante a $\log Z$, lasciando il gradiente in $\theta$ invariato e migliorando la stabilità numerica (codice `demo_algorithm.py:368`).

**Q. Mostri che il reward GAIL $-\log D$ deriva dall'obiettivo.**
La policy minimizza $\mathbb E_\pi[\log D(s,a)]$; massimizzare $-\log D$ è equivalente, quindi $r=-\log D$. Nel codice `softplus(-\text{logit})=-\log\sigma(\text{logit})=-\log D$.

**Q. Perché in AIRL il logit del discriminatore è $f-\log\pi$?**
Da $D=\frac{e^f}{e^f+\pi}=\sigma(f-\log\pi)$, quindi $\text{logit}\,D=f-\log\pi$ e il reward $\log D-\log(1-D)=f-\log\pi$ (codice `reward_nets.py:425-428`).

### 8.3 Domande implementative

**Q. Come entra il reward appreso in PPO?**
`EnvRewardWrapper.step_wait` sostituisce il reward d'ambiente con `reward_model.predict(...)`, normalizzato a media 0/std 1 con una running stat (`env_wrappers.py:80-93`).

**Q. Come si garantisce che la mistura di importance copra le policy giuste?**
`_model_buffer` e `_policy_window` sono due `deque` con lo stesso `maxlen` che evictano in lock-step; gli snapshot sono presi in `push_data` quando la policy è ancora quella che ha generato il rollout (`demo_algorithm.py:117-118,146-147`).

**Q. Perché GAIL/AIRL ereditano da DemoAlgorithm se non usano la sua loss?**
Per riusare l'outer loop, il buffering e l'infrastruttura; sovrascrivono `_compute_reward_loss` e annullano `_update_importance_weights`. (È anche la fonte del buffer inutilizzato, §3.5 p.4.)

**Q. Dove sono gestite le transizioni terminali in AIRL?**
Il fattore $(1-\text{done})$ azzera $V(s')$ (`reward_nets.py:414`); per transizioni terminali senza $s'$ si usa $s'=0$ come placeholder sicuro (`airl_algorithm.py:129-131`).

### 8.4 Domande critiche

**Q. La sua implementazione di AIRL gode del disentanglement del Teorema 5.1?**
No: `reward_net` usa $(s,a)$, non solo $s$. Recupero un reward valido per imitazione nello stesso MDP, ma **non garantito trasferibile** a dinamiche diverse. Per ottenere il disentanglement dovrei restringere $g$ a $g(s)$.

**Q. Il reward GAIL ha survival bias: come lo controlla?**
Il reward $-\log D>0$ favorisce episodi lunghi; lo mitigo con `ent_coef` e con la centratura della normalizzazione, ma il problema è strutturale e proprio per questo ho implementato AIRL (che centra il logit, `before_agent_training`).

**Q. Pesare le dimostrazioni sotto la mistura di policy è corretto?**
È un'approssimazione: la vera distribuzione generatrice delle demo non è la policy. Se la policy diverge molto, l'estimatore di $Z$ diventa rumoroso. È dichiarato nel codice e mitigato dal centering e dalla finestra di fusione.

**Q. Con `fragment_length=1`, le preferenze su segmenti di Christiano degenerano?**
Sì, a confronti per-transizione: si perde la struttura temporale. È legittimo per un reward markoviano denso come questo task, ma è una semplificazione rispetto al paper.

**Q. La normalizzazione cumulativa del reward non crea non-stazionarietà?**
Sì: `_RunningMeanStd` accumula su tutto il training, quindi reagisce lentamente ai cambiamenti del reward model tra iterazioni. Una media a finestra o un reset periodico sarebbero più aderenti allo spirito di Christiano §2.2.

---

## Appendice A — Iperparametri effettivi (dai config)

| Iperparam. | Chri | Demo | GAIL | AIRL |
|---|---|---|---|---|
| `lr_rew` | 3e-4 | 1e-4 | 1e-3 | 1e-3 |
| `gradient_steps_rew` | 100 | 100 | 100 | 100 |
| batch | 128 (pref) | 32E/64M | 32E/64M | 64E/64M |
| `l2_rew` | 0.0 | 0.01 | 0.01 | 0.01 |
| `fragment_length` | 1 | 100 | 1 | 1 |
| `n_ensembles` | 3 | 3 | 3 | 3 |
| `ent_coef` (PPO) | 0 | 0 | 0.01 | 0.01 |
| `gamma` | 0.997 | 0.997 | 0.997 | 0.997 (anche shaping) |
| `total_timesteps` | 2e6 | 2e6 | 4e6 | 2e6 |
| `timesteps_per_iteration` | 20000 | 20000 | 20000 | 20000 |
| specifici | `fragmenter:active`, `labels:binary_bernulli`, `T=20`, `initial_queries=1000` | `policy_window=10`, `grad_clip=1.0` | — | `bc_warmup=0` |

## Appendice B — Componenti condivisi (file chiave)

- **Outer loop:** `base_reward_learning_algorithm.py::train` (`:407-468`).
- **Reward → PPO:** `env_wrappers.py::EnvRewardWrapper`.
- **Buffering traiettorie:** `env_wrappers.py::EnvBufferingWrapper`.
- **Reti reward/discriminatore:** `reward_nets.py` (`SumoRewardNet`, `StateActionDiscriminatorNet`, `AirlRewardNet`, `NormalizedRewardNet`, `GailRewardNet`, `RewardEnsemble`).
- **Oracoli sintetici:** `gatherers.py` (`PreferenceGathererFromReward`, `DemoGathererFromExpert`).
- **Selezione query:** `fragmenters.py` (`RandomPairFragmenter`, `HighVariancePairFragmenter`).
- **Tipi:** `types.py` (`Transition`, `Trajectory`, `Fragment`, `FragmentPair`, `Preference`).
