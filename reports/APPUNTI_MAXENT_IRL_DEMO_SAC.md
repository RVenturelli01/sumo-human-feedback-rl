# Appunti — Reward learning da dimostrazioni (MaxEnt IRL) + agente SAC

Trascrizione delle conversazioni di analisi sull'implementazione in
[`human-feedback-rl/human_feedback_rl/`](human-feedback-rl/human_feedback_rl/).
Algoritmo: `DemoAlgorithm` (MaxEnt IRL) che addestra un reward model dalle
dimostrazioni esperte e allena un agente SAC sul reward appreso.

---

## 0. Come funziona la pipeline (ricostruzione oggettiva)

Loop alternato in [`demo_algorithm.py:174-208`](human-feedback-rl/human_feedback_rl/algorithms/demo_algorithm.py#L174-L208):

1. **Bootstrap** (opzionale): raccoglie `initial_agent_timesteps`, allena il
   reward, fa pre-warm di SAC.
2. Per iterazione: campiona rollout dell'agente (da `rollout_env` dedicato) →
   allena il reward model → rinormalizza il reward per l'agente → allena SAC per
   `timesteps_per_iteration`.
3. Reward model = **ensemble di MLP** che mappa
   `(obs, action, next_status_onehot[7], done) → scalare`
   ([`reward_nets.py:103-130`](human-feedback-rl/human_feedback_rl/common/reward_nets.py#L103-L130)).
4. SAC consuma il reward appreso via `EnvRewardWrapper` e, opzionalmente, via
   relabeling del replay buffer.

**Dati esperti verificati:** 3050 traiettorie, lunghezza media ~164, range
**1–401**, `next_status` 7-dim presente, terminali = 2723 arrived / 327 collided,
`log_policy_prob=None` (corretto: l'esperto non è il proposal).

---

## 1. Le loss e perché solo `maxent_corrected` è teoricamente corretta

Le 5 loss in [`losses.py:41-67`](human-feedback-rl/human_feedback_rl/algorithms/demo/losses.py#L41-L67):

| Loss | Formula | Giudizio |
|------|---------|----------|
| `maxent_corrected` | `-E_exp[r/τ] + logsumexp(r/τ − log q) − log N` | **Unica stimatore MaxEnt consistente** (importance sampling corretto) |
| `maxent` | `-E_exp[r] + logsumexp(r_model) − log N` | **Biased**: omette il peso `1/q`. Corretto solo se il proposal è uniforme |
| `maxent_2` | come sopra ma partition su esperti+modello | Stesso bias |
| `demo`/`demo_loss` | `-E_exp[r] + E_model[r]` | Differenza di medie, **non limitata inferiormente**; minimo solo grazie all'L2 |
| `demo_corrected` | `softplus(-(score_exp−score_model)/τ)` ranking su reward medio/step | Limitata, ragionevole |

**Nota pratica:** la config di default usa `maxent`, il launcher usa `maxent_2`
→ in produzione si gira la variante **biased**.

### Idea di fondo della MaxEnt IRL

L'esperto si comporta come se scegliesse le traiettorie con probabilità
proporzionale a `exp(reward)`:

```
p(τ) = exp(r(τ)) / Z       con  Z = Σ_τ exp(r(τ))  (partition function)
```

La loss (negative log-likelihood) è:

```
loss = −r(esperto) + log Z
```

- 1° termine: **alza** il reward sulle traiettorie esperte.
- 2° termine (`log Z`): **abbassa** il reward su tutte le altre.

Tutto il problema è stimare `Z`, somma su infinite traiettorie.

### Il punto chiave: il campione è sbilanciato

Non campioniamo traiettorie uniformemente: le campioniamo con la **policy
dell'agente** `q`. Serve la correzione **importance sampling**:

```
Z ≈ media sui campioni di [ exp(r(τ)) / q(τ) ]
```

Il `/ q(τ)` "sconta" le traiettorie che l'agente produce spesso e "amplifica"
quelle rare. Senza, stai assumendo `q` uniforme — falso.

- `maxent_corrected` include `− log q` → corregge → **stima corretta**.
- `maxent`/`maxent_2` non lo includono → assumono `q` uniforme → **stima distorta**
  (tanto più grave quanto più la policy è lontana dall'uniforme).
- `demo` non stima nemmeno `Z`: alza/abbassa il reward "in media"; non ha minimo,
  lo tiene a bada solo l'L2 → scala del reward arbitraria.

> Tensione teoria/pratica: `maxent_corrected` è corretto ma **più rumoroso**
> (l'importance sampling esplode quando `q` è piccolo → per questo si monitora
> l'*effective sample size*). Per stabilità molti usano comunque le versioni biased.

---

## 2. Derivazione: `Z ≈ (1/N)Σ exp(r/τ)/q` → `logsumexp(r/τ − log q) − log N`

Sono **equivalenti**, solo riscrittura algebrica.

**Trasformazione 1** — portare `/q` dentro l'esponenziale (`q = exp(log q)`):

```
exp(r/τ) / q  =  exp(r/τ) / exp(log q)  =  exp(r/τ − log q)
```

**Trasformazione 2** — `log(Σ exp) = logsumexp`. Con `aᵢ = r(τᵢ)/τ − log q(τᵢ)`:

```
log Z ≈ log( (1/N) Σ exp(aᵢ) )
      = log( Σ exp(aᵢ) ) − log N
      = logsumexp_i( r(τᵢ)/τ − log q(τᵢ) ) − log N
```

Che è esattamente [`losses.py:65-66`](human-feedback-rl/human_feedback_rl/algorithms/demo/losses.py#L65-L66):

```python
corrected_logits = model_returns / self.temperature - log_q
partition = th.logsumexp(corrected_logits, dim=0) - np.log(len(model_returns))
```

**Perché tenere la forma logsumexp** (non `log(Σ exp(r)/q)`):
1. **Stabilità numerica**: `exp(somma di reward su 400 passi)` fa overflow;
   `logsumexp` usa il trucco del max e non esplode.
2. **`log q` è già in forma log** (`Σ_t log π(a_t|s_t)`); usare `q` grezzo
   richiederebbe `exp(log q)` → underflow su traiettorie lunghe.

---

## 3. Cosa succede a convergenza (agente imita l'esperto)

"Convergenza" = il reward smette di cambiare **e** `q` ha raggiunto la
distribuzione MaxEnt `p ∝ exp(r/τ)`.

### 3.1 Le importance weight si appiattiscono (auto-consistenza)

Pesi: `wᵢ = exp(rᵢ/τ − log qᵢ)`. A convergenza `log q = r/τ − log Z`, quindi:

```
rᵢ/τ − log qᵢ = log Z   (costante per tutti)
```

- **Tutti i pesi diventano uguali** → ESS → N (massimo), stima di `Z` a varianza
  zero. L'agente che imita l'esperto diventa il **proposal perfetto** per stimare `Z`.
- Osservabile: `effective_sample_fraction` → 1
  ([`reward_diagnostics.py:110`](human-feedback-rl/human_feedback_rl/algorithms/demo/reward_diagnostics.py#L110)).

### 3.2 Feature/occupancy matching

Gradiente: `∇loss = −E_esperto[∇r] + E_p[∇r]`. Si annulla quando:

```
E_esperto[feature] = E_agente[feature]   (occupancy matching)
```

Osservabili:
- `reward/expert_model_margin` → 0.
- `imitation/state_action_auc` → 0.5 (il classificatore non distingue più
  esperto da agente) — segnale pulito di imitazione riuscita.

### 3.3 Cosa NON succede

- **Reward non unico**: a gradiente zero esiste un'intera famiglia di reward con
  la stessa occupancy (ambiguità del reward). L'L2 sceglie quale.
- **Agente non deterministico**: MaxEnt = massima entropia → riproduce anche la
  *stocasticità* dell'esperto. Sinergia naturale col termine entropia di SAC.

### 3.4 Note critiche per questo setup

- Tutto vale **solo per `maxent_corrected`**. Con `maxent`/`maxent_2` i pesi sono
  `softmax(r)` invece di `softmax(r − log q)` → non si appiattiscono → punto fisso
  **spostato** (agente sovra-concentra sui modi ad alto reward, AUC può non
  tornare a 0.5).
- La **rinormalizzazione per-iterazione** del reward
  ([`reward_training.py:73`](human-feedback-rl/human_feedback_rl/algorithms/demo/reward_training.py#L73))
  rende la convergenza un bersaglio mobile.

---

## 4. `log Z` è il minimo della loss? NO

`log Z` è solo **uno dei due termini**, non il minimo.

### Valore della loss al minimo

A convergenza `p_θ = p_esperto`. Per una traiettoria esperto:
`r(τ)/τ = log p(τ) + log Z`. Quindi:

```
E_esperto[r/τ] = E_esperto[log p] + log Z = −H(p) + log Z
loss = −(−H(p) + log Z) + log Z = H(p)
```

**Il `log Z` si cancella.** Il minimo della loss è l'**entropia della
distribuzione esperta** `H(p)`, non `log Z`.

### Perché `log Z` non può essere un minimo

La loss è **invariante** allo shift `r → r + c`:
- `−E_esperto[r/τ]` diminuisce di `c/τ`,
- `log Z` aumenta di `c/τ` → si annullano.

Ma `log Z` da solo cambia di `c/τ`, con `c` arbitrario → `log Z` può valere
qualsiasi cosa → non può essere il minimo. (È il volto matematico
dell'ambiguità del reward: scala/offset non identificabili.)

### Conseguenze pratiche

1. **Non giudicare la convergenza dal valore assoluto di `reward/loss` o `log Z`**:
   dominati da una costante arbitraria + rinormalizzazione.
2. **Guarda le quantità invarianti**: `margin → 0`, `AUC → 0.5`, `ESS → 1`.
3. La loss è limitata inferiormente da `H(esperto) ≥ 0` (non va a −∞, a differenza
   di `demo` che non ha minimo e dipende interamente dall'L2).

---

## 5. A convergenza il gradiente si annulla? SÌ, ma in valore atteso

`∇loss = −E_esperto[∇r] + E_p[∇r]` → zero quando le occupancy coincidono.
Quattro asterischi:

1. **Zero solo in media.** Il gradiente reale è stimato da minibatch finiti →
   `Var(∇̂) > 0`. I parametri **non si congelano**, tremolano. `reward/grad_norm`
   ([`reward_training.py:61`](human-feedback-rl/human_feedback_rl/algorithms/demo/reward_training.py#L61))
   scende a un *pavimento di rumore*, non a 0.

2. **Il weight decay non si annulla.** Gradiente totale `= ∇loss + 2λθ`; nullo solo
   dove `∇loss = −2λθ` (punto regolarizzato), non dove `∇loss = 0`. L'L2 sceglie il
   reward tra quelli equivalenti.

3. **Equilibrio accoppiato, non un minimo.** Due giocatori (reward separa, SAC
   imita); serve gradiente nullo per **entrambi** insieme → tipo Nash. Convergenza
   non garantita, può **oscillare**. In più ogni iterazione fa solo
   `gradient_steps_rew` passi sul reward → lo zero emerge dal loop, non da una
   singola chiamata.

4. **Con loss biased lo zero è nel posto sbagliato.** `grad_norm → 0` può accadere
   a un punto che NON è occupancy matching. "Gradiente nullo" ≠ "imitazione corretta".

### Sintesi operativa

| | Comportamento a convergenza |
|---|---|
| Gradiente reward atteso | → 0 (feature matching), **solo con `maxent_corrected`** |
| `reward/grad_norm` osservato | → piccolo floor, non 0 (rumore minibatch + L2) |
| Parametri reward | jitter attorno all'equilibrio |
| Sistema reward+SAC | equilibrio accoppiato, può oscillare |

**Segnale di convergenza sana**: `grad_norm` basso e stabile **insieme** a
`margin → 0` e `AUC → 0.5`. Se `grad_norm` basso ma AUC ancora alta → punto fisso
spurio (tipico delle loss biased).

---

## 6. Altri rilievi dall'analisi (da approfondire)

- **Bias di lunghezza**: la famiglia `maxent` usa return = **somma non scontata**
  su traiettorie 1–401 passi; `exp(somma)` nella partition è dominato dalle più
  lunghe. `demo_corrected` normalizza per lunghezza, `maxent*` no.
- **Mismatch di discount**: reward appreso su return non scontate, SAC ottimizza
  con `gamma=0.995–0.997`.
- **Doppia normalizzazione ridondante** in
  [`reward_nets.py:295-307`](human-feedback-rl/human_feedback_rl/common/reward_nets.py#L295-L307):
  i normalizzatori per-membro non vengono mai settati (identità).
- **Rischio reward-hacking via status**: reward condizionato su `next_status` →
  può collassare in un classificatore dello status terminale (arrived vs collided).
- **`relabel_rewards`**: True nella config, False nel launcher → con False il
  critic SAC allena su reward stale di reward model più vecchi.
- **Minori**: `ThreadPoolExecutor(max_workers=1)` rende seriale l'allenamento
  "parallelo" dell'ensemble; `DemonstrationDataset` non usato da questo algoritmo.

### Punti aperti da verificare

1. Uso di `maxent`/`maxent_2` (biased) invece di `maxent_corrected`: scelta
   consapevole (stabilità/costo) o default da rivedere?
2. Bias di lunghezza sulle somme non scontate: accettato come approssimazione?
3. Su quale asse approfondire: correttezza matematica delle loss / integrazione
   SAC+relabeling / reward-hacking via status?

---

## 7. Diagnosi di un run reale (`maxent_corrected`, ~477k step) — agente 68% offroad

> ⚠️ **Diagnosi iniziale, parzialmente rivista (vedi §9).** Le osservazioni
> *loggate* (ESS=1, top1≈1, reward_offroad≈neutro) sono fatti. Ma il **nesso
> causale** "ESS=1 → 68% offroad" è rimasto **un'ipotesi non dimostrata**:
> su più run il comportamento NON segue in modo pulito l'ESS. Leggi questa
> sezione come una lettura meccanica di *uno snapshot*, non come causa accertata.

Run con `loss_type=maxent_corrected`, `temperature=20`, `batch_size_model=256`,
`gradient_steps_rew=15`, `normalize_agent_reward=true`, `relabel_rewards=true`.
Esito: `off_road=0.68`, `collisions=0.17`, `successes=0.15`, `ep_len_mean=48`
(esperto arrived ≈ 179).

### La partition è degenerata: ESS = 1

```
maxent_corrected_effective_sample_fraction = 0.0039
maxent_corrected_effective_sample_size     = 1.00
maxent_corrected_top1_softmax_weight       = 0.99999
maxent_corrected_log_q_mean                = 74.1
```

Il gradiente della partition `Σ_i w_i ∇r(τ_i)` con un solo `w ≈ 1` collassa su
**una sola traiettoria**. Tutte le altre traiettorie agente — comprese quelle
offroad — ricevono gradiente ≈ 0. Il canale che dovrebbe penalizzare l'offroad
è di fatto **spento**.

**Perché τ=20 non basta:** `temperature` scala solo `r/τ`, non `log q`. La
varianza dominante viene da `log q` (somma di 48–400 log-prob → decine di
spread), che τ non tocca. È la maledizione dell'IS su orizzonte lungo.

### Correzione importante a un'imprecisione mia

Non serve che l'esperto vada offroad: in teoria la partition (sui campioni
agente) abbassa l'offroad. Il vero problema è che **con ESS=1 la partition non
penalizza nulla tranne una traiettoria**. I due canali per l'offroad sono
entrambi morti: termine esperto (offroad assente, 0/500088) **e** termine
partition (degenerato). Per questo `reward_offroad = −1.05` (quasi neutro)
mentre `reward_collided = −12.8` (collided **è** nei dati esperti → riceve
gradiente dal termine esperto).

### Reward quasi piatto → niente shaping

```
gap_arrived_running = 0.018      # raggiungere il goal ≈ girovagare
```

Tra gli stati running non c'è struttura: SAC non ha gradiente per imparare a
tenere la corsia → deriva offroad.

---

## 8. Fix implementato: partition a frammenti per `maxent_corrected` (strategia 1)

Idea: fare l'importance sampling su **finestre di `k` passi** invece che su
traiettorie intere. `log q` su `k` passi ha varianza ~(L/k) volte più piccola →
i pesi non collassano → ESS sale → la partition torna a penalizzare l'offroad.
La formula resta identica; cambia solo *cosa* sono `r` e `log q` (somme su `k`
passi). Vedi sezione "mostra ad alto livello" della discussione.

**Scelta di design (strategia 1):** frammenti isolati a `maxent_corrected`
(loss **e** diagnostica), le altre 5 loss intatte.

Modifiche:
- [`losses.py`](human-feedback-rl/human_feedback_rl/algorithms/demo/losses.py):
  `_reward_loss` instrada `maxent_corrected` a `_maxent_corrected_loss`; nuovi
  helper `_fragment_returns`, `_fragment_log_probs` (log q per frammento,
  **allineato** ai return), `_traj_step_rewards`/`_traj_step_log_probs`.
  `_fragment_step` con `fragment_length=None` → frammento = traiettoria intera.
- [`reward_diagnostics.py`](human-feedback-rl/human_feedback_rl/algorithms/demo/reward_diagnostics.py):
  blocco diagnostico `maxent_corrected` frammentato (usa `len(partition_logits)`
  per il `−log N`), così partition/ESS/loss loggati = quelli ottimizzati.
- [`demo_algorithm.py`](human-feedback-rl/human_feedback_rl/algorithms/demo_algorithm.py):
  parametro `fragment_length: Optional[int] = None` con validazione.
- Config/launcher: `fragment_length: null` esposto.

**Cosa è verificato (confidenza ALTA):**
- Equivalenza: `fragment_length=None` riproduce **al bit** la loss storica a
  traiettoria intera (`allclose` True) → nessuna regressione.
- `log q` allineato ai return per frammento.
- Non tocca le altre 5 loss (isolamento strategia 1).

**Cosa NON è verificato (confidenza BASSA — vedi §9):**
- Che i frammenti siano un fix *teoricamente fondato*. I frammenti **non sono
  campioni iid** da un proposal: sono pezzi consecutivi della stessa traiettoria.
  La proprietà di consistenza dell'importance sampling che giustificava
  `maxent_corrected` **non si trasferisce automaticamente** al livello di
  frammento. È un obiettivo *diverso* (MaxEnt "locale"), nello spirito di
  GCL/AIRL, ma non dimostrato corretto.
- Che i frammenti **aiutino in pratica**. Le run con frammenti (k=10, k=30)
  continuano a oscillare e a far collassare l'ESS a tratti. Non esiste un A/B
  controllato (None vs 10 vs 30, stesso seed/stadio) che lo dimostri.

---

## 9. Stato onesto: oscillazione del success-rate — cosa so e cosa no

Sintomo osservato su **tutte** le run analizzate (k=10, k=30, l2=0.1): il
success-rate **oscilla** (picchi 0.72–0.89, poi crolli a ~0 con
collisioni/offroad), senza convergere.

### Ipotesi scartate / ridimensionate

| Ipotesi (mia, in corso d'opera) | Stato | Prova |
|---|---|---|
| **La scala del return** causa il collasso ESS / l'oscillazione | ❌ **SMENTITA** | `Spearman(ret_std, ESS)` ≈ 0 e segno incoerente su 3 run (−0.18, +0.13, −0.07) |
| Il **collasso ESS** causa l'oscillazione | ❌ **non dimostrato** | mai verificato il nesso ESS↔comportamento |
| `l2_rew` alto stabilizza | ❌ **smentito** | run l2=0.1: `ret_std` cresce a ~600, oscilla comunque |
| I **frammenti** alzano stabilmente l'ESS / risolvono | ⚠️ **incerto** | nessun A/B; le run oscillano comunque |

### Ipotesi attuale (la meglio argomentata)

**Causa più probabile: instabilità strutturale del loop alternato reward↔agente
(obiettivo non stazionario).** Il reward definisce il bersaglio guardando dove sta
l'agente *adesso*; l'agente lo insegue; quando ci arriva, il reward lo ridefinisce
e l'agente (policy unica che si riscrive) perde il comportamento buono.

**Confidenza: MODERATA (~60%).**
- A favore: è una proprietà nota dell'IRL avversariale (best-response che cicla);
  l'oscillazione è **robusta** a tutti i cambi di iperparametro provati.
- Contro: **non isolata** con un esperimento; meccanismo esatto non provato.

### L'esperimento decisivo (non ancora fatto)

**Congelare il reward model** dopo N iterazioni e lasciare SAC ottimizzare sul
reward fisso:
- se l'agente **converge** → è il reward mobile la causa (confidenza → alta);
- se **oscilla comunque** → la causa è (anche) l'instabilità di SAC.

È l'unico modo per trasformare il ~60% in una risposta vera.

### Cosa resta solido di tutto il documento

Le sezioni **1–5** (teoria MaxEnt IRL: derivazione, convergenza, `log Z` non è il
minimo, gradiente a convergenza) e le osservazioni **loggate** restano valide.
Le **attribuzioni causali** delle §7–8 (scala/ESS → comportamento) sono ipotesi,
non fatti.
