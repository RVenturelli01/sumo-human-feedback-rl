# Riassunti — Inverse RL & Reward Learning da feedback umano

Tre paper chiave sull'apprendimento di funzioni di ricompensa (reward) a partire dal
comportamento o dal feedback umano, dal più classico al più recente:

1. **Ziebart et al. (2008)** — *Maximum Entropy IRL* → come risolvere l'ambiguità dell'IRL con un principio probabilistico.
2. **Boularias et al. (2011)** — *Relative Entropy IRL* → come fare la stessa cosa **senza conoscere il modello dell'ambiente**.
3. **Baur et al. (2026)** — *MAVRL* → come imparare **un unico reward** da **tanti tipi di feedback diversi** insieme, in modo bayesiano.

---

## 0. Contesto: cos'è l'Inverse Reinforcement Learning (IRL) e perché è difficile

Nel **Reinforcement Learning (RL) classico** conosciamo la ricompensa e cerchiamo la
politica (policy) migliore. Nell'**Inverse RL** facciamo il contrario: osserviamo un
esperto che si comporta bene e vogliamo scoprire **quale ricompensa** stava (implicitamente)
massimizzando. Serve perché spesso è più facile *mostrare* un comportamento desiderato che
*scrivere a mano* la funzione di reward corretta (guidare bene, camminare, essere "utile e
sicuro", ecc.).

Il problema però è **mal posto (ill-posed)**, per due motivi:

- **Ambiguità**: tantissime funzioni di reward diverse spiegano lo stesso comportamento
  (inclusa la reward costante zero, che rende ottimale *qualsiasi* politica). Serve un
  criterio per scegliere *quale* reward preferire tra le infinite compatibili.
- **Imperfezione umana**: le dimostrazioni reali sono rumorose e sub-ottimali. Nessun singolo
  reward le spiega alla perfezione, quindi non possiamo pretendere un match esatto.

L'idea che unifica tutti e tre i paper è: **modellare il comportamento in modo probabilistico**.
Invece di assumere che l'esperto sia perfetto, si assume che scelga azioni/traiettorie *buone*
con probabilità più alta, ma non deterministicamente. Questo trasforma l'IRL in un problema
di **stima statistica** ben definito.

---

## 1. Ziebart et al. (2008) — *Maximum Entropy Inverse Reinforcement Learning*

### Il problema che risolve
I metodi IRL precedenti (feature matching di Abbeel & Ng, 2004) chiedevano che la politica
appresa producesse le stesse **feature medie** dell'esperto. La reward è assunta lineare nelle
feature:

$$R(\zeta) = \theta^\top \mathbf{f}_\zeta = \sum_{s_j \in \zeta} \theta^\top \mathbf{f}_{s_j}$$

dove $\mathbf{f}_\zeta$ è la somma delle feature (es. "numero di semafori", "km su autostrada")
lungo la traiettoria $\zeta$, e $\theta$ sono i pesi da imparare.

Il vincolo di **feature matching** è:

$$\sum_{\zeta} P(\zeta)\, \mathbf{f}_\zeta = \tilde{\mathbf{f}} \quad (\text{= media empirica dell'esperto})$$

Il problema: **moltissime distribuzioni** $P(\zeta)$ soddisfano questo vincolo. Quale scegliere?
I metodi precedenti sceglievano in modo arbitrario, introducendo un bias non giustificato.

### L'idea centrale: massima entropia
Ziebart applica il **principio di massima entropia** (Jaynes): tra tutte le distribuzioni che
rispettano i dati osservati (il feature matching), scegli **quella con entropia massima**, cioè
la più "indecisa"/uniforme possibile. È l'unica scelta che **non aggiunge nessuna assunzione**
oltre a ciò che i dati impongono. Ogni altra distribuzione starebbe implicitamente "inventandosi"
informazione che non abbiamo osservato.

Massimizzare l'entropia sotto il vincolo di feature matching dà una soluzione in forma chiusa:
una **distribuzione esponenziale sulle traiettorie**.

$$\boxed{\;P(\zeta \mid \theta) = \frac{1}{Z(\theta)} \exp\!\left(\theta^\top \mathbf{f}_\zeta\right)\;}$$

**Come leggerla, in parole semplici:**
- Traiettorie con reward più alto sono **esponenzialmente più probabili**.
- Ma due traiettorie con lo *stesso* reward sono **equiprobabili** (nessuna preferenza arbitraria).
- $Z(\theta)$ è la costante di normalizzazione (*partition function*), la somma su tutte le traiettorie.

Questo è essenzialmente un modello di **razionalità di Boltzmann**: l'esperto è "razionale ma
rumoroso", tanto più probabile a fare la cosa giusta quanto più questa è migliore delle alternative.

### Caso stocastico
Se l'ambiente ha transizioni casuali, la probabilità di una traiettoria dipende anche dalla
dinamica $T$. Gli autori usano un'approssimazione che condiziona sulla partition function, in modo
che l'agente **non venga "premiato" per la fortuna** (per esiti casuali favorevoli fuori dal suo
controllo):

$$P(\zeta \mid \theta, T) \approx \frac{\exp(\theta^\top \mathbf{f}_\zeta)}{Z(\theta, T)}
\prod_{s_{t+1},a_t,s_t \in \zeta} P_T(s_{t+1}\mid a_t, s_t)$$

### Come si imparano i pesi $\theta$
Si massimizza la **verosimiglianza** (log-likelihood) delle traiettorie dimostrate:

$$\theta^* = \arg\max_\theta \sum_{\text{esempi}} \log P(\zeta \mid \theta)$$

La funzione è **concava** (nel caso deterministico), quindi ha un unico ottimo. Il gradiente ha
una forma molto intuitiva:

$$\nabla L(\theta) = \underbrace{\tilde{\mathbf{f}}}_{\text{feature dell'esperto}}
- \underbrace{\sum_{s_i} D_{s_i}\, \mathbf{f}_{s_i}}_{\text{feature attese dal modello}}$$

Cioè: **aggiusta i pesi finché le feature attese sotto il modello uguagliano quelle
dell'esperto**. Qui $D_{s_i}$ è la **frequenza attesa di visita** dello stato $s_i$ (quanto spesso,
in media, il modello passa per quello stato).

### L'algoritmo (forward pass)
Il pezzo tecnico è calcolare le frequenze di visita $D_{s_i}$ **senza enumerare tutte le
traiettorie** (che sono esponenzialmente tante). Si usa una programmazione dinamica tipo
*forward-backward*:

1. **Backward pass**: calcola un "soft value" di ogni stato — come il value RL classico, ma con
   `softmax` (log-sum-exp) al posto di `max`. Da qui la policy locale $P(a\mid s) \propto \exp(\cdot)$.
2. **Forward pass**: partendo dalla distribuzione iniziale degli stati, propaga in avanti nel tempo
   quanto spesso si visita ogni stato, seguendo quella policy.

### Esperimenti — predizione di percorsi in auto
L'applicazione celebre: dati GPS reali di **25 tassisti a Pittsburgh** (>100.000 miglia). Il modello:
- predice la **destinazione** anche osservando solo un pezzo del tragitto;
- predice **quale strada** sceglierà il guidatore;
- cattura preferenze reali (evitare code, semafori, strade lente).

Batte i baseline (modelli di Markov, feature matching deterministico) perché gestisce bene il
comportamento rumoroso e sub-ottimale.

### In sintesi
- **Contributo**: risolve l'ambiguità dell'IRL con un principio pulito (max entropy = zero bias
  arbitrario) e fornisce un **modello generativo** del comportamento, non solo una policy.
- È la **base concettuale** di quasi tutto l'IRL moderno (Deep MaxEnt IRL, GAIL, e — importante per
  te — è lo stesso schema Boltzmann-razionale che ricompare nella *demonstration likelihood* di MAVRL).
- **Limite pratico**: richiede di **conoscere la dinamica** dell'ambiente e di risolvere ripetutamente
  il problema forward (costoso o impossibile in ambienti sconosciuti/continui). È esattamente
  questo limite che il paper successivo attacca.

---

## 2. Boularias, Kober & Peters (2011) — *Relative Entropy Inverse Reinforcement Learning*

### Il problema che risolve
MaxEnt IRL ha un tallone d'Achille: **serve il modello di transizione** dell'ambiente e bisogna
risolvere ripetutamente un problema di RL "in avanti". In robotica ad alta dimensione, il modello
spesso non c'è. Boularias et al. propongono una versione **model-free**: nessuna conoscenza né
stima della dinamica, basta poter **campionare traiettorie**.

### L'idea centrale: minimizzare l'entropia relativa (KL)
Invece di massimizzare l'entropia in assoluto, si **minimizza la divergenza di Kullback-Leibler
(KL, o "entropia relativa")** tra la distribuzione appresa $P(\tau)$ e una **distribuzione di
riferimento** $Q(\tau)$ (una policy di baseline nota da cui sappiamo campionare, per esempio
casuale/uniforme), sempre sotto il vincolo di feature matching:

$$\min_{P}\; \sum_\tau P(\tau)\, \ln\frac{P(\tau)}{Q(\tau)}$$

soggetto a:

$$\Big| \sum_\tau P(\tau)\,\mathbf{f}_\tau - \hat{\mathbf{f}}^{\pi_E}\Big| \le \varepsilon
\qquad\text{e}\qquad \sum_\tau P(\tau) = 1$$

**Intuizione:** "resta il più vicino possibile alla baseline $Q$, cambiando solo quel tanto che
basta per riprodurre le feature dell'esperto". Il vincolo con tolleranza $\varepsilon$ ammette
esplicitamente che il match non sia esatto (dati finiti e rumorosi).

**Relazione con MaxEnt:** se $Q$ è uniforme, minimizzare la KL da $Q$ **equivale** a massimizzare
l'entropia. Quindi RE-IRL **generalizza** MaxEnt: permette in più di iniettare conoscenza a priori
scegliendo una $Q$ informativa.

### La forma della soluzione
Come in MaxEnt, la soluzione ottima è esponenziale, ma **"agganciata" alla baseline** $Q$:

$$P(\tau) = \frac{Q(\tau)\,\exp(\theta^\top \mathbf{f}_\tau)}
{\sum_{\tau'} Q(\tau')\,\exp(\theta^\top \mathbf{f}_{\tau'})}$$

I pesi $\theta$ sono i **moltiplicatori di Lagrange** del vincolo di feature matching, e si trovano
ottimizzando il **problema duale**, che è concavo (un solo ottimo).

### Il trucco che la rende model-free: importance sampling
Il punto cruciale. La normalizzazione e le feature attese sotto $P$ **non si possono calcolare in
forma chiusa** (servirebbe il modello). Si **stimano per campionamento** usando traiettorie generate
dalla baseline $Q$ — che sappiamo simulare o raccogliere:

$$\sum_\tau P(\tau)\,\mathbf{f}_\tau \;\approx\;
\frac{\sum_i \exp(\theta^\top \mathbf{f}_{\tau_i})\, \mathbf{f}_{\tau_i}}
{\sum_i \exp(\theta^\top \mathbf{f}_{\tau_i})}, \qquad \tau_i \sim Q$$

Ogni traiettoria campionata da $Q$ viene **ripesata** per il fattore $\exp(\theta^\top \mathbf{f})$
(questo è l'*importance sampling*). Risultato: **la dinamica dell'ambiente non compare mai** nelle
formule. Servono solo campioni: quelli dell'esperto (per $\hat{\mathbf{f}}^{\pi_E}$) e quelli della
baseline (per stimare il resto).

### Ottimizzazione
Si usa una salita del **(sub)gradiente** sul duale. Il gradiente mantiene la forma familiare
"feature dell'esperto − feature attese", con le seconde ottenute per importance sampling. Gli autori
discutono due varianti: model-free puro, e una versione che corregge per le probabilità di
transizione quando queste sono parzialmente note (riduce la varianza).

### Esperimenti
- **Gridworld** benchmark: RE-IRL raggiunge prestazioni **comparabili a MaxEnt senza conoscere il
  modello**.
- Problemi di controllo tipo **racetrack** e task simil-robotici: recupera reward sensati usando
  solo traiettorie campionate.

### In sintesi
- **Contributo**: primo IRL **model-free** pratico, via entropia relativa + importance sampling;
  generalizza MaxEnt (che si riottiene con $Q$ uniforme) e consente prior informativi.
- **Limite**: la qualità della stima dipende **fortemente** da quanto la baseline $Q$ è vicina alla
  distribuzione target. Se $Q$ è mal scelta o lo spazio è grande, la varianza dell'importance
  sampling esplode e servono moltissimi campioni.

---

## 3. Baur et al. (2026) — *MAVRL: Learning Reward Functions from Multiple Feedback Types with Amortized Variational Inference*

### Il problema che risolve
Il feedback umano arriva in **forme molto diverse**: dimostrazioni, confronti/preferenze
(A meglio di B), valutazioni scalari (voti 1–5), interventi tipo "stop" (fermare l'agente quando
sbaglia). Ogni tipo dà informazione **parziale e complementare**:

- **Dimostrazioni**: precise *lungo* il percorso dell'esperto, ma non dicono nulla sul resto dello
  spazio degli stati.
- **Preferenze**: coprono più zone, ma danno solo informazione *relativa* (A > B), senza intensità
  né valori assoluti; ne servono tantissime.
- **Valutazioni/rating**: identificano bene gli stati-obiettivo, ma sono informazione limitata e
  ordinale.
- **Stop**: dicono forte e chiaro *cosa evitare* (zone pericolose/scadenti), ma quasi nulla su cosa
  sia *desiderabile*.

I metodi esistenti o addestrano **un reward separato per tipo** e li combinano *post hoc*
(problema: scale e incertezze diverse, difficili da riconciliare), oppure **collassano tutto in una
sola modalità** (es. convertono tutto in preferenze), perdendo l'informazione specifica di ciascun
tipo. Bilanciare a mano i pesi delle loss delle diverse modalità è arbitrario e fragile.

### L'idea centrale: un unico reward latente, inferenza bayesiana
MAVRL tratta la reward $R$ come una **variabile latente condivisa** e ogni tipo di feedback come una
**osservazione probabilistica** di quella stessa reward. Si vuole il **posterior bayesiano**:

$$p(R \mid \mathcal{D}) = \frac{p(\mathcal{D}\mid R)\, p(R)}{\int p(\mathcal{D}\mid R')\,p(R')\,dR'}$$

Il punto chiave: dato $R$, le diverse modalità di feedback sono **condizionatamente indipendenti**,
quindi la likelihood **fattorizza** in modo pulito:

$$p(\mathcal{D}\mid R) = \prod_m p(\mathcal{D}^{(m)}\mid R)$$

In parole: ogni tipo di feedback $m$ contribuisce **il proprio termine di likelihood**, tutti riferiti
alla *stessa* reward. Non serve un rappresentazione intermedia comune né pesi di bilanciamento a
mano — è la struttura probabilistica a combinare le informazioni.

### Le likelihood specifiche per tipo (il cuore modellistico)
Ogni modalità è un modo diverso di "guardare" la reward $R$ (o i suoi Q-value $Q_R^*$):

- **Preferenze → Bradley-Terry.** La probabilità che $\xi_1$ sia preferita a $\xi_2$:
  $$p(\xi_1 \succ \xi_2 \mid R) = \frac{\exp(\beta R(\xi_1))}{\exp(\beta R(\xi_1)) + \exp(\beta R(\xi_2))}$$
  con $\beta$ = temperatura inversa (quanto sono "decise" le scelte umane). *(È la stessa likelihood
  usata nel tuo ramo preferenze / Christiano PPO.)*

- **Dimostrazioni → policy Boltzmann-razionale** (l'eredità diretta di MaxEnt IRL):
  $$p(\xi \mid R) = \prod_{(s,a)\in\xi} \frac{\exp(\beta Q_R^*(s,a))}{\sum_b \exp(\beta Q_R^*(s,b))}$$
  L'esperto sceglie azioni con Q-value alto, con rumore controllato da $\beta$.

- **Rating → regressione ordinale (ordered logit).** Ogni traiettoria ha un'utilità latente sotto
  $R$; il voto discreto $y\in\{1,\dots,K\}$ nasce da soglie (*cutpoints*) $\psi_k$ non equispaziate
  (gli umani non usano scale lineari):
  $$p(y=k\mid \xi,R) = F(\psi_k - R(\xi)) - F(\psi_{k-1} - R(\xi))$$
  con $F$ = sigmoide logistica. Nota: **non** è regressione diretta sul reward — è solo informazione
  ordinale, senza giudizio assoluto.

- **Stop → modello di hazard a tempo discreto.** Lo stop segnala che il comportamento è degradato
  oltre il tollerabile. Si definisce una "sub-ottimalità istantanea"
  $\Delta_R(s,a) = \max_b Q_R^*(s,b) - Q_R^*(s,a)$ e un hazard (rischio di intervento) che cresce con
  la sub-ottimalità accumulata (scontata all'indietro con $\rho$):
  $$h_R^{\lambda,\rho}(\xi,\tau) = 1 - \exp\!\Big(-\lambda \sum_{t=1}^{\tau}\rho^{\tau-t}\Delta_R(s_t,a_t)\Big)$$
  Il tempo di stop segue poi una distribuzione geometrica ("nessuno stop fino a $\tau$, poi stop");
  se non c'è stop, l'osservazione è *right-censored*. Più grande $\lambda$, più "severo" il supervisore.

**Punto di forza estensibile:** qualsiasi nuovo tipo di feedback (ranking, correzioni, ...) si
aggiunge semplicemente **definendone la likelihood**, senza toccare il resto dell'architettura.

### Perché serve l'inferenza variazionale (e "amortized")
Il posterior esatto (Eq. 1) è **doppiamente intrattabile**: bisognerebbe integrare sia sulle scelte
di feedback sia su tutte le possibili reward. MCMC (l'approccio bayesiano "gold standard") funziona
ma è lentissimo e non scala.

La **Variational Inference (VI)** aggira il problema: approssima il posterior con una distribuzione
più semplice $q_\theta(R)$, ottimizzando l'**Evidence Lower Bound (ELBO)**:

$$\mathbb{E}_{R\sim q_\theta}[\log p(\mathcal{D}\mid R)] - D_{\mathrm{KL}}(q_\theta(R)\,\|\,p(R))$$

Il primo termine massimizza la verosimiglianza dei dati; il secondo tiene $q_\theta$ vicino al prior
(regolarizzazione). **"Amortized"** (stile Variational Autoencoder) significa: invece di ri-ottimizzare
per ogni input, si addestra una **rete neurale encoder** che mappa direttamente coppie
stato-azione $(s,a)$ a una distribuzione sulla reward. Il *reparameterization trick*
($R = \mu + \sigma\odot\epsilon$, con $\epsilon\sim\mathcal{N}(0,I)$) fa fluire i gradienti attraverso
il campionamento. Costruiscono sul framework **AVRIL** (Chan & van der Schaar, 2021), sostituendo la
sua singola *demonstration likelihood* con **l'intero set** di likelihood per-tipo.

### L'architettura MAVRL, in concreto
- Un **reward encoder** $q_\theta(R\mid s,a) = \mathcal{N}(\mu_\theta(s,a), \sigma^2_\theta(s,a))$:
  reti neurali che producono **media e varianza** della reward per ogni $(s,a)$. La varianza dà
  **incertezza interpretabile** (vedi sotto).
- Una **funzione azione-valore ausiliaria** $Q_\phi(s,a)$, necessaria per le likelihood che
  dipendono dai Q-value (dimostrazioni e stop).
- Un **termine di consistenza TD (temporal-difference)** che lega reward e $Q_\phi$ tramite Bellman
  a un passo ($\delta_\phi = Q_\phi(s,a) - \gamma Q_\phi(s',a')$), tenendo il reward coerente con il
  suo value.

**Obiettivo unificato** (un solo ELBO per tutte le modalità):

$$\mathcal{L}_{\text{MAVRL}} = \sum_{m=1}^{M}
\mathbb{E}_{\mathcal{D}^{(m)}}\Big[\mathbb{E}_{R\sim q_\theta}[\log p_\psi^{(m)}(y\mid R,Q_\phi)]\Big]
- \lambda_{\mathrm{KL}}\, D_{\mathrm{KL}}(q_\theta\|p) + \lambda_{\mathrm{TD}}\,\mathcal{L}_{\mathrm{TD}}$$

Ogni tipo di feedback entra come **un termine di log-likelihood** nella stessa somma: niente pesi di
bilanciamento cross-modali arbitrari. Restano solo due categorie di iperparametri: quelli *fisici*
del modello di annotatore (temperatura $\beta$, cutpoints $\psi_k$, hazard $\lambda,\rho$ — hanno
significato interpretabile) e i due coefficienti di regolarizzazione $\lambda_{\mathrm{KL}},
\lambda_{\mathrm{TD}}$. L'addestramento è **asincrono e invariante all'ordine**: i mini-batch delle
diverse modalità arrivano indipendentemente e si combinano liberamente (Algoritmo 1).

### Risultati sperimentali principali
Testato su griglie tabulari (`grid_cliff`, `grid_sparse`, `grid_trap`) e controllo continuo
(Acrobot, CartPole, LunarLander), con dimostrazioni, preferenze, rating e stop, isolati e combinati.

1. **I tipi di feedback si complementano** (Fig. 1): ciascuno produce una struttura di reward e di
   incertezza caratteristica — dimostrazioni sicure lungo il percorso ma cieche altrove, preferenze
   ampie ma imprecise, rating focalizzati sull'obiettivo, stop bravi a marcare le zone da evitare.
   **Combinandoli** si ricostruisce il reward vero con alta fedeltà.

2. **Combinare migliora policy e recupero del reward** (Tab. 1): la combinazione di **tutte** le
   modalità è la migliore o quasi in 4 ambienti su 6, e ottiene la **EPIC distance** più bassa (metrica
   di quanto il reward appreso è vicino a quello vero, a meno di shaping) nei domini tabulari. Nessuna
   singola modalità domina ovunque (le dimostrazioni brillano nei reward sparsi, i rating in CartPole).

3. **Gli stop complementano le altre modalità**: aggiungere gli stop migliora quasi sempre le
   combinazioni, perché portano informazione sulla **sub-ottimalità** difficile da ottenere altrove.

4. **Efficiente vs baseline non-variazionali** (Tab. 2): a parità di prestazioni, MAVRL è **~30×**
   più veloce di MCMC in addestramento e — a differenza di MCMC — resta trattabile nel continuo
   (niente risoluzione MDP nel loop interno). Batte nettamente il *Post-Hoc Averaging* (media di reward
   addestrati separatamente), confermando che i reward separati confondono scale diverse e perdono la
   complementarità cross-modale.

5. **Più robusto sotto perturbazioni** (Fig. 2, Tab. 3): quando si perturba la dinamica al deployment
   (più stocasticità, gravità/vento diversi) o si corrompe il feedback (annotatori più rumorosi del
   previsto), i reward da **multi-feedback degradano molto più dolcemente** dei singoli tipi. Effetto di
   **compensazione**: per 3 corruzioni su 4 (preferenze, rating, stop) la combinazione mantiene ≥90%
   della prestazione, mentre i baseline a singola modalità crollano a ratio di 0.10. **Eccezione**: le
   **dimostrazioni** restano il punto debole strutturale — corromperle trascina giù anche la combinazione.

### Limiti (dichiarati dagli autori)
- Valutazione solo su **feedback simulato**, non su umani veri (che hanno bias sistematici e giudizi
  dipendenti dal contesto).
- L'**incertezza** del reward viene inferita ma **non ancora sfruttata** per guidare la raccolta di
  feedback — l'**active learning** su feedback eterogeneo è lasciato come lavoro futuro.

### In sintesi
- **Contributo**: primo framework che fa **inferenza bayesiana congiunta** su un **unico reward
  latente** da **feedback eterogeneo**, con un solo obiettivo variazionale, senza bilanciamento manuale
  delle loss e senza collassare le modalità. Scalabile (amortized VI) ed estensibile (basta definire una
  nuova likelihood).
- **Eredità diretta**: la *demonstration likelihood* è la policy Boltzmann-razionale di **MaxEnt IRL**;
  la *preference likelihood* è Bradley-Terry (come nel filone Christiano/RLHF).

---

## 4. Il filo conduttore (come si legano i tre paper)

| Aspetto | **MaxEnt IRL** (2008) | **Relative Entropy IRL** (2011) | **MAVRL** (2026) |
|---|---|---|---|
| Cosa impara | Reward da **dimostrazioni** | Reward da **dimostrazioni** | Reward da **feedback multiplo** (demo, pref, rating, stop) |
| Principio | Massima entropia | Minima entropia relativa (KL) da baseline $Q$ | Inferenza bayesiana (VI / ELBO) |
| Modello dell'ambiente | **Richiesto** (forward DP esatto) | **Non richiesto** (model-free) | Q-value ausiliario appreso + consistenza TD |
| Come stima le attese | Programmazione dinamica esatta | **Importance sampling** da $Q$ | Rete encoder amortizzata + reparameterization |
| Distribuzione | $P(\tau)\propto e^{\theta^\top f_\tau}$ | $P(\tau)\propto Q(\tau)\,e^{\theta^\top f_\tau}$ | posterior $q_\theta(R)$ gaussiano per $(s,a)$ |
| Incertezza | No (stima puntuale) | No | **Sì** (varianza del posterior, interpretabile) |
| Punto debole | Serve il modello, costoso | Varianza dell'importance sampling | Solo feedback simulato; incertezza non ancora sfruttata |

**In una frase:** tutti e tre spiegano il comportamento con una **distribuzione esponenziale sul
reward** per risolvere l'ambiguità dell'IRL. MaxEnt lo fa esattamente ma **serve il modello**; RE-IRL
lo rende **model-free** con l'importance sampling; MAVRL generalizza il tutto a **molti tipi di
feedback insieme** in un quadro **bayesiano scalabile**, dove ogni modalità è semplicemente una diversa
*likelihood* sullo stesso reward latente.

**Rilevanza per il tuo lavoro (hybrid demo+preference RL):** MAVRL è la formalizzazione "pulita" del
problema che stai affrontando con il ramo ibrido demo+preferenze. Il conflitto di scala che hai
osservato (`[[hybrid-reward-scale-conflict]]`) è proprio ciò che MAVRL evita **non** combinando loss
con pesi a mano, ma facendo entrare demo (likelihood Boltzmann/MaxEnt) e preferenze (Bradley-Terry)
come termini di log-likelihood sullo **stesso** reward latente, con la coerenza garantita dal termine
TD sui Q-value invece che da un bilanciamento manuale.
