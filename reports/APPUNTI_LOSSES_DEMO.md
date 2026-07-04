# Appunti sulle loss di `demo/losses.py`

File analizzato: `human-feedback-rl/human_feedback_rl/algorithms/demo/losses.py`.

## 1. Contesto e notazione

Il modulo contiene le loss usate da `DemoAlgorithm` per addestrare il reward
model a partire da due insiemi di traiettorie:

- traiettorie esperte, campionate da `self.expert_trajectories`;
- traiettorie del modello/agente, campionate da `self.trajectories`.

Indichiamo con:

- $\tau_e$ una traiettoria esperta e con $\tau_m$ una traiettoria generata
  dall'agente;
- $r_\theta(s_t,a_t,s_{t+1},d_t)$ il reward prodotto dalla rete;
- $R_\theta(\tau)=\sum_t r_\theta(s_t,a_t,s_{t+1},d_t)$ il return appreso
  dell'intera traiettoria;
- $N_e$ e $N_m$ il numero di traiettorie esperte e modello nel batch;
- $T_i=|\tau_i|$ la lunghezza di una traiettoria;
- $q(\tau)$ la probabilità della traiettoria sotto la policy che ha generato
  i rollout;
- $\beta>0$ il parametro `temperature` del codice.

Il training minimizza la loss. Di conseguenza un termine
$-\mathbb{E}[R_\theta(\tau_e)]$ spinge verso l'alto il reward degli esperti.

I nomi accettati sono:

```text
maxent
maxent_2
demo
demo_loss
maxent_corrected
demo_corrected
```

`demo` e `demo_loss` sono due alias della stessa identica loss. In pratica ci
sono quindi cinque obiettivi distinti.

## 2. Campionamento e calcolo dei return

`_sample_trajectories` campiona senza reinserimento i due batch. Se il dataset
contiene meno elementi del batch size richiesto, usa tutti gli elementi:

$$
N_e=\min(\texttt{batch\_size\_expert},|D_e|),\qquad
N_m=\min(\texttt{batch\_size\_model},|D_m|).
$$

`_traj_step_rewards` ricostruisce tensori contenenti osservazione, azione,
stato successivo e flag `done`, poi invoca il reward model. `_traj_sum_reward`
somma i reward per-step senza interrompere il grafo computazionale: i gradienti
possono quindi propagarsi dalla loss fino ai parametri del reward model.

Nota importante: quasi tutte le loss storiche lavorano sul **return totale**.
Una traiettoria lunga contiene più termini e può quindi ricevere un valore
assoluto maggiore solo per effetto della durata. `demo_corrected` è l'unica
variante che confronta esplicitamente reward medi per step.

## 3. `demo` e `demo_loss`

### Formula

```python
expert_term = -expert_returns.mean()
loss = expert_term + model_returns.mean()
```

ovvero:

$$
\mathcal{L}_{demo}
=-\frac{1}{N_e}\sum_i R_\theta(\tau_{e,i})
+\frac{1}{N_m}\sum_j R_\theta(\tau_{m,j}).
$$

Minimizzarla significa aumentare il return degli esperti e diminuire quello
delle traiettorie prodotte dall'agente. È un confronto lineare tra le due
medie, non una vera likelihood MaxEnt.

### Proprietà e criticità

- È semplice e produce gradienti su tutte le traiettorie con peso uniforme.
- È invariante all'aggiunta della stessa costante a tutti i return del batch.
- Non usa `temperature`: cambiarla non modifica questa loss.
- Non è limitata inferiormente. Se il reward model riesce a separare esperto e
  agente, può continuare ad amplificare indefinitamente i reward positivi e
  negativi. Nel progetto il `weight_decay=l2_rew` dell'ottimizzatore aiuta a
  contenere questa deriva, ma non cambia la natura della loss.
- Il confronto tra return totali introduce un possibile bias di lunghezza.
- Non esegue pairing fra una specifica traiettoria esperta e una del modello:
  confronta soltanto le medie dei due gruppi.

## 4. `demo_corrected`

### Formula

Le prime $n=\min(N_e,N_m)$ traiettorie dei due batch vengono accoppiate per
indice. Per ogni coppia viene calcolato il margine tra reward medi per step:

$$
m_i=
\frac{R_\theta(\tau_{e,i})}{T_{e,i}}
-\frac{R_\theta(\tau_{m,i})}{T_{m,i}}.
$$

La loss è:

$$
\mathcal{L}_{demo\_corrected}
=\frac{1}{n}\sum_i
\operatorname{softplus}\left(-\frac{m_i}{\beta}\right)
=\frac{1}{n}\sum_i
\log\left(1+e^{-m_i/\beta}\right).
$$

È una loss logistica di ranking: è piccola quando il reward medio per step
dell'esperto supera quello del modello.

### Ruolo della temperatura

- Una temperatura piccola rende la transizione attorno a $m=0$ più netta e
  aumenta la scala del gradiente.
- Una temperatura grande rende il confronto più morbido e riduce il gradiente.

### Proprietà e criticità

- Elimina il bias più evidente dovuto alla diversa lunghezza delle traiettorie.
- Penalizza soprattutto coppie con margine negativo o piccolo; le coppie già
  ben ordinate ricevono gradiente progressivamente minore.
- La loss è non negativa e tende a zero per margini positivi molto grandi.
- Il pairing per indice non ha un significato semantico: poiché entrambi i batch
  sono campionati casualmente, equivale a costruire coppie casuali indipendenti.
- Se i batch hanno dimensioni diverse, usa soltanto le prime
  $\min(N_e,N_m)$ traiettorie; gli elementi eccedenti non contribuiscono.
- Normalizzare per lunghezza cambia l'obiettivo: può essere desiderabile per
  confrontare la qualità media delle azioni, ma rimuove l'informazione sulla
  durata totale dell'episodio.

## 5. `maxent`

### Formula implementata

```python
loss = (
    -expert_returns.mean()
    + torch.logsumexp(model_returns, dim=0)
    - np.log(len(model_returns))
)
```

ovvero:

$$
\mathcal{L}_{maxent}
=-\frac{1}{N_e}\sum_i R_\theta(\tau_{e,i})
+\log\left(\frac{1}{N_m}\sum_j e^{R_\theta(\tau_{m,j})}\right).
$$

Il secondo termine è una `log-mean-exp` dei return del modello e funge da
surrogato empirico della log-partition function.

### Interpretazione del gradiente

La derivata del termine di partizione rispetto a ciascun return modello è una
softmax:

$$
w_j=\frac{e^{R_j}}{\sum_k e^{R_k}}.
$$

Le traiettorie del modello con reward più alto ricevono quindi la penalizzazione
maggiore. La loss prova ad alzare i return esperti e ad abbassare soprattutto le
traiettorie non esperte che il reward model considera migliori.

### Perché è una formula “storica” e non una NLL MaxEnt completa

Se le traiettorie modello sono campionate da una policy $q$, la media
implementata stima:

$$
\mathbb{E}_{\tau\sim q}[e^{R_\theta(\tau)}],
$$

non direttamente la partition function rispetto alla misura desiderata. Manca
la correzione per la proposal $q(\tau)$. Quindi il risultato dipende da quali
traiettorie la policy corrente visita e con quale frequenza. La variante
`maxent_corrected` introduce questa correzione.

Altre osservazioni:

- `temperature` è ignorata anche da questa loss storica;
- `logsumexp` rende il calcolo numericamente più stabile di
  `log(exp(x).sum())`;
- pochi return modello molto grandi possono concentrare quasi tutto il peso
  softmax, causando alta varianza e bassa effective sample size (ESS);
- la loss usa return totali ed è quindi sensibile alla lunghezza;
- sottrarre `log(N_m)` trasforma `logsumexp` in una log-media e rende la scala
  meno direttamente dipendente dalla dimensione del batch.

## 6. `maxent_2`

### Formula

Questa variante inserisce nella partizione sia le traiettorie modello sia le
traiettorie esperte:

$$
\mathcal{L}_{maxent\_2}
=-\frac{1}{N_e}\sum_iR_\theta(\tau_{e,i})
+\log\left(
\frac{
\sum_j e^{R_\theta(\tau_{m,j})}
+\sum_i e^{R_\theta(\tau_{e,i})}
}{N_m+N_e}
\right).
$$

### Differenza rispetto a `maxent`

Gli esperti ricevono due contributi di gradiente opposti:

1. il primo termine aumenta uniformemente tutti i loro return;
2. il termine di partizione penalizza attraverso la softmax soprattutto le
   traiettorie, anche esperte, con return maggiore.

Questo può regolarizzare i valori estremi degli esperti, ma rende
l'interpretazione meno pulita: il set positivo viene usato contemporaneamente
come dato da premiare e come parte della normalizzazione. Anche qui non compare
la correzione per la distribuzione di campionamento e `temperature` non viene
usata.

## 7. `maxent_corrected`

### Formula a traiettoria intera

Con `fragment_length=None`, ogni traiettoria costituisce un unico elemento. Il
codice calcola:

$$
z_j=\frac{R_\theta(\tau_{m,j})}{\beta}-\log q(\tau_{m,j})
$$

e poi:

$$
\mathcal{L}_{maxent\_corrected}
=-\frac{1}{N_e}\sum_i\frac{R_\theta(\tau_{e,i})}{\beta}
+\log\left(\frac{1}{N_m}\sum_j e^{z_j}\right).
$$

Poiché

$$
e^{z_j}=\frac{e^{R_\theta(\tau_{m,j})/\beta}}{q(\tau_{m,j})},
$$

il termine di partizione è un estimatore importance-sampling. A differenza di
`maxent`, compensa il fatto che i rollout non siano campionati uniformemente ma
da una proposal policy $q$.

### Calcolo di `log q`

La probabilità logaritmica della traiettoria è la somma delle log-probabilità
delle azioni:

$$
\log q(\tau)=\sum_t \log\pi(a_t\mid s_t).
$$

Il codice preferisce i valori `log_policy_prob` salvati nelle transizioni al
momento del rollout. Se non sono tutti disponibili, li ricalcola con
`policy_action_log_probs(self.agent, obs, actions)`.

I `log q` vengono convertiti in numeri/tensori senza gradiente: sono pesi della
proposal, non una parte da ottimizzare assieme al reward model.

`DemoAlgorithm` richiede un `rollout_env` dedicato quando viene scelta questa
loss. L'intento è raccogliere traiettorie da una proposal policy coerente senza
desincronizzare l'ambiente di training SB3. Ricalcolare le probabilità in un
secondo momento con una policy ormai aggiornata sarebbe invece meno affidabile;
quando possibile è preferibile conservarle durante il rollout.

### Temperatura

Qui `temperature` viene applicata coerentemente sia al termine esperto sia ai
return nella partizione. Non scala invece `log q`, come richiesto dalla formula
di importance sampling.

### Varianza dei pesi

La correzione è teoricamente più motivata, ma può avere alta varianza. Una
traiettoria con $q(\tau)$ molto piccola produce un grande valore
$-\log q(\tau)$ e può dominare la `logsumexp`. Nei log di diagnostica il codice
calcola per questo motivo anche:

$$
ESS=\frac{1}{\sum_j w_j^2},\qquad
w=\operatorname{softmax}(z).
$$

Una effective sample fraction `ESS / numero_elementi` inferiore a `0.1` genera
un warning. Aumentare il batch modello migliora la copertura, ma non garantisce
da solo pesi ben bilanciati.

## 8. Frammentazione in `maxent_corrected`

`fragment_length` viene usata **solo** da `maxent_corrected`.

- `None`: una traiettoria intera è un frammento;
- intero positivo $K$: ogni traiettoria viene divisa in finestre consecutive
  di massimo $K$ step; l'ultima può essere più corta.

Per ogni frammento vengono sommati separatamente reward e log-probabilità, così
i due vettori restano allineati. Per esempio, una traiettoria di 5 step con
`fragment_length=2` produce frammenti di lunghezza 2, 2 e 1.

Questa opzione può ridurre l'ampiezza e la varianza di `log q`, ma **cambia la
loss**. I frammenti consecutivi dello stesso rollout non sono campioni i.i.d. da
una proposal fissa. L'argomento di consistenza dell'importance sampling a
livello di traiettoria non si trasferisce automaticamente alle finestre.

Il commento nel sorgente è quindi corretto nel presentare
`fragment_length > 0` come obiettivo MaxEnt locale/windowed, vicino nello
spirito a GCL/AIRL, ma sperimentale e non come “versione più corretta” della
loss. Inoltre traiettorie più lunghe generano più frammenti e acquistano
implicitamente più peso nella media.

## 9. Confronto sintetico

| Loss | Confronto | Temperatura | Correzione `q` | Gestione lunghezza | Rischio principale |
|---|---|---:|---:|---|---|
| `demo` / `demo_loss` | differenza tra medie | no | no | return totale | scala del reward non limitata |
| `demo_corrected` | ranking logistico a coppie | sì | no | reward medio per step | pairing casuale; scarta il batch eccedente |
| `maxent` | esperto + partizione sui rollout | no | no | return totale | partizione dipendente dalla proposal |
| `maxent_2` | esperto + partizione su entrambi i set | no | no | return totale | interpretazione mista degli esperti |
| `maxent_corrected` | NLL MaxEnt con importance sampling | sì | sì | intera o frammenti | alta varianza dei pesi / bassa ESS |

## 10. Considerazioni pratiche

### Invarianza a uno shift dei return

Tutte le formule sono invarianti se si aggiunge la stessa costante al **return**
di ogni elemento coinvolto: il contributo negativo esperto e quello positivo
del modello si cancellano. Questo non equivale necessariamente ad aggiungere
una costante al reward per step quando le traiettorie hanno lunghezze diverse,
perché lo shift del return diventerebbe proporzionale alla durata.

### Scala del reward

Le loss storiche non usano `temperature`. La scala viene quindi controllata
principalmente dall'ottimizzazione e dalla regolarizzazione L2. La
normalizzazione del reward destinato all'agente, gestita altrove da
`_update_agent_reward_normalization`, non altera il forward grezzo usato per
allenare queste loss.

### Stabilità numerica

Il codice usa correttamente `torch.logsumexp` e `softplus`, entrambe forme
stabili. Il training controlla inoltre che loss e norma del gradiente siano
finite prima dell'update. Rimane però possibile una forte concentrazione dei
pesi softmax; per le loss MaxEnt è quindi utile monitorare ESS, peso top-1,
massa top-5, range e deviazione standard dei return.

### Precondizioni implicite

Quando `_reward_loss` viene chiamata devono esistere almeno una traiettoria
esperta e una del modello; altrimenti `stack`, `mean` o `logsumexp` lavorerebbero
su collezioni vuote. Il costruttore valida il dataset esperto e il training evita
l'update se `self.trajectories` è vuoto.

## 11. Lettura consigliata delle varianti

- Usare `demo` come baseline storica semplice, tenendo sotto controllo scala dei
  reward e differenze di durata.
- Preferire `demo_corrected` se l'obiettivo è un ranking robusto della qualità
  media per step e non serve una specifica interpretazione probabilistica
  MaxEnt.
- Considerare `maxent` e `maxent_2` come surrogate storiche utili per confronti
  e ablation, non come stimatori importance-corrected della partition function.
- Usare `maxent_corrected` quando si vogliono rispettare le probabilità della
  proposal, verificando che i log-prob siano quelli del rollout e monitorando
  attentamente l'ESS.
- Trattare `fragment_length > 0` come esperimento su un obiettivo locale distinto
  e confrontarlo esplicitamente con la baseline teoricamente coerente
  `fragment_length=None`.

## 12. Valutazione rispetto alla letteratura

Non esiste una loss universalmente corretta: la scelta dipende da quale modello
probabilistico si assume per le dimostrazioni e da cosa si vuole ottenere dal
reward. Tuttavia, se l'obiettivo dichiarato è **Maximum Entropy IRL**, alcune
loss di questo modulo sono teoricamente molto più giustificate di altre.

### 12.1 Classificazione generale

| Loss | Valutazione teorica | Uso consigliato |
|---|---|---|
| `maxent_corrected`, traiettorie intere | la più vicina alla likelihood MaxEnt corretta | obiettivo principale per MaxEnt IRL |
| `demo_corrected` | valida come ranking/preference surrogate | alternativa pratica, senza presentarla come MaxEnt IRL |
| `maxent_2` | euristica non consistente come likelihood MaxEnt | baseline o ablation |
| `maxent` | manca la correzione per la proposal | baseline storica |
| `demo` / `demo_loss` | confronto lineare potenzialmente illimitato | solo baseline semplice |

### 12.2 Perché `maxent_corrected` è la variante più fondata

Nella MaxEnt IRL classica, una traiettoria ha una distribuzione della forma:

$$
p_\theta(\tau)
=\frac{1}{Z_\theta}
\exp\left(\frac{R_\theta(\tau)}{\beta}\right).
$$

Se le traiettorie usate per stimare la partition function sono generate da una
proposal policy $q$, allora:

$$
Z_\theta
=\mathbb{E}_{\tau\sim q}
\left[
\frac{\exp(R_\theta(\tau)/\beta)}{q(\tau)}
\right].
$$

Questo produce esattamente i logits implementati dalla loss:

$$
\frac{R_\theta(\tau)}{\beta}-\log q(\tau).
$$

La struttura deriva dalla formulazione probabilistica della
[Maximum Entropy IRL di Ziebart et al.](https://23.aaai.org/Library/AAAI/2008/aaai08-227.php)
e dalla sua approssimazione sample-based in
[Guided Cost Learning](https://proceedings.mlr.press/v48/finn16.html). In
particolare, Guided Cost Learning mostra che eliminare gli importance weights
produce in generale una stima inconsistente della likelihood.

La correttezza richiede però alcune ipotesi:

- `fragment_length=None`, in modo che ogni campione sia una traiettoria intera;
- i `log_policy_prob` devono appartenere alla policy che ha realmente prodotto
  ciascuna azione;
- esperto e agente devono condividere dinamica e distribuzione iniziale, oppure
  le differenze devono essere incluse nel rapporto di importance sampling;
- la proposal deve avere supporto sulle traiettorie rilevanti;
- le densità devono essere calcolate rispetto alla stessa misura delle azioni;
- i rollout raccolti da proposal diverse devono conservare il rispettivo
  denominatore, o essere trattati mediante una mixture/fusion distribution.

Nel codice corrente `self.trajectories` viene sostituito a ogni iterazione e le
log-probabilità vengono salvate durante la raccolta. Questo evita il problema
più grave di ricalcolare tutto con una policy successiva già modificata.

È importante distinguere tra stimatore della partition function e log della
partition function. La media importance-sampling può stimare $Z_\theta$, ma:

$$
\mathbb{E}[\log \hat Z_\theta]
\neq
\log Z_\theta
$$

con un numero finito di campioni. Di conseguenza la loss e il suo gradiente
possono avere bias finito-campione, pur risultando consistenti quando il numero
di campioni cresce e le condizioni dell'importance sampling sono soddisfatte.

**Verdetto:** è la variante teoricamente preferibile per dichiarare un
esperimento di MaxEnt IRL, ma può essere difficile da ottimizzare a causa della
varianza dei pesi.

### 12.3 Perché la frammentazione non è automaticamente corretta

Con `fragment_length > 0`, finestre consecutive della stessa traiettoria vengono
trattate come elementi separati della partition function. Non è una conseguenza
diretta della likelihood a traiettoria intera perché:

- i frammenti dello stesso rollout sono correlati;
- la distribuzione dello stato iniziale di ogni frammento è indotta dalla
  proposal e dal comportamento precedente;
- le traiettorie più lunghe generano più campioni e ricevono più peso;
- il codice non include una densità esplicita dello stato iniziale del
  frammento né una derivazione condizionata del relativo obiettivo.

Metodi come [AIRL](https://arxiv.org/abs/1710.11248) lavorano a livello di
transizioni tramite un obiettivo avversariale derivato appositamente. Spezzare
una likelihood di traiettoria non rende da solo la loss equivalente ad AIRL.

**Verdetto:** `fragment_length > 0` è un'euristica locale sperimentale. Può
ridurre la varianza, ma non deve essere presentata come una versione più
corretta dell'importance-sampling MaxEnt.

### 12.4 `maxent`: utile come surrogata, non come likelihood MaxEnt

La partizione di `maxent` calcola:

$$
\log\left(
\frac{1}{N_m}
\sum_j \exp(R_\theta(\tau_j))
\right),
\qquad \tau_j\sim q.
$$

Questa media stima:

$$
\mathbb{E}_{\tau\sim q}
[\exp(R_\theta(\tau))],
$$

non la partition function rispetto alla misura di traiettoria desiderata. La
formula coinciderebbe con quella corretta soltanto in casi particolari, per
esempio se $q$ fosse esattamente la misura base. Questo è generalmente falso
per una policy SAC continua.

La loss può comunque funzionare come obiettivo contrastivo: alza i return degli
esperti e penalizza soprattutto le traiettorie dell'agente alle quali il reward
model assegna valori elevati. Il risultato dipende però dalla distribuzione dei
negativi visitati dalla policy.

**Verdetto:** non è una likelihood MaxEnt consistente; è accettabile come
surrogata storica o baseline.

### 12.5 `maxent_2`: stabilizzazione euristica

Guided Cost Learning osserva che aggiungere dimostrazioni al background sample
set può impedire che l'obiettivo diventi illimitato con batch piccoli. Tuttavia,
nella derivazione sample-based i campioni devono ancora essere associati alle
rispettive densità proposal, spesso attraverso una fusion distribution.

`maxent_2` condivide l'idea pratica di inserire gli esperti nella partizione, ma
non usa i relativi importance weights. Inoltre ogni esperto riceve due gradienti
opposti:

1. il termine $-\mathbb{E}[R_e]$ ne aumenta il return;
2. la softmax nella partition function ne penalizza il return, soprattutto se è
   già elevato.

Questo può limitare valori estremi e migliorare la stabilità numerica, ma non
costituisce una stima consistente della likelihood MaxEnt.

**Verdetto:** euristica plausibile e potenzialmente stabile, da usare per
baseline o ablation. Un run con `LOSS_TYPE=maxent_2` non è un run MaxEnt IRL
importance-corrected.

### 12.6 `demo`: problema di scala e illimitatezza

Con un reward model sufficientemente espressivo, la loss:

$$
-\mathbb{E}[R_\theta(\tau_e)]
+\mathbb{E}[R_\theta(\tau_m)]
$$

può continuare a diminuire aumentando i reward esperti e diminuendo quelli del
modello. Non contiene una normalizzazione probabilistica, una saturazione, un
margine o un vincolo esplicito sulla funzione reward.

Il weight decay rende il problema finito più gestibile, ma lega la scala e la
soluzione alla forza della regolarizzazione. L'obiettivo può essere interpretato
come un semplice separatore tra due distribuzioni, non come una likelihood IRL.

**Verdetto:** non consigliata come loss principale; utile come baseline minima.

### 12.7 `demo_corrected`: corretta sotto un modello di preferenza

La loss per coppia può essere riscritta come:

$$
-\log\sigma\left(
\frac{\bar R_e-\bar R_m}{\beta}
\right),
$$

dove $\bar R$ è il reward medio per step. È una normale loss logistica di
ranking, collegata al modello Bradley--Terry. Obiettivi analoghi vengono usati
da [T-REX](https://proceedings.mlr.press/v97/brown19a.html) per imparare reward
da traiettorie ordinate.

La formula è ben motivata se il dato significa realmente che la traiettoria
esperta è preferibile a quella del modello. Nel codice questa etichetta viene
assegnata automaticamente a ogni coppia casuale. L'assunzione diventa meno
credibile quando l'agente raggiunge o supera l'esperto: la loss continuerà
comunque a imporre $\bar R_e>\bar R_m$.

La normalizzazione per lunghezza elimina parte del duration bias, ma cambia la
semantica del confronto. Il reward medio per step non è sempre equivalente al
return totale: dipende dal fatto che il task premi qualità media, successo,
rapidità o permanenza nell'episodio.

**Verdetto:** buona loss pratica e relativamente stabile se interpretata come
preference/ranking learning, non come Maximum Entropy IRL.

### 12.8 Raccomandazione per gli esperimenti

Per un esperimento principale presentato come MaxEnt IRL:

1. usare `maxent_corrected`;
2. impostare `fragment_length=None`;
3. iniziare con `exploration_frac=0.0` per semplificare la proposal;
4. usare le log-probabilità memorizzate durante il rollout;
5. usare un batch modello abbastanza ampio;
6. monitorare ESS, effective sample fraction, peso top-1, massa top-5 e range
   dei corrected logits;
7. ripetere il training su più seed, perché un singolo run può essere dominato
   dalla varianza degli importance weights.

Come confronti:

- `demo_corrected` è l'alternativa pratica basata sul ranking;
- `maxent`, `maxent_2` e `demo` sono baseline storiche/ablation;
- `maxent_corrected` con frammenti è un'ablation separata sull'obiettivo locale.

Se `maxent_corrected` è instabile, non segue automaticamente che la teoria sia
sbagliata: il problema può essere la varianza dell'importance sampling o una
proposal con copertura insufficiente. Una soluzione più robusta a livello di
transizione, come AIRL, richiederebbe però una modifica algoritmica sostanziale
e non soltanto un cambio di loss.

### 12.9 Livello di confidenza

**Confidenza complessiva: 90%.**

La confidenza è alta sulla classificazione teorica delle formule rispetto a
MaxEnt IRL, importance sampling e ranking probabilistico. È più bassa sulla
previsione di quale loss dia empiricamente il risultato migliore nello
specifico ambiente SUMO: questo dipende dalla qualità delle dimostrazioni,
dalla copertura della proposal, dalla distribuzione delle durate e dalla
varianza effettiva dei log-pesi.

## 13. Diagnosi di `maxent_corrected` con SAC nelle run sul server

Questa sezione analizza i sintomi osservati nelle run sul server senza assumere
i valori presenti nelle configurazioni locali:

- `return_std` fino a circa 1000;
- margine esperto--modello nell'ordine di 1000--2000;
- `state_action_auc > 0.9`;
- success rate molto basso, con prevalenza di episodi off-road brevi;
- `effective_sample_fraction < 0.01`;
- norma del gradiente che oscilla tra valori elevati, circa 80, e valori vicini
  a zero;
- con frammenti di lunghezza fissa, success rate oscillante e ESS fraction
  ancora molto bassa.

La spiegazione più probabile non è un singolo bug, ma una catena causale:

1. l'importance sampling collassa;
2. il reward model impara a separare esperto e agente senza identificare
   necessariamente il task desiderato;
3. SAC sfrutta una scorciatoia basata sulla terminazione off-road;
4. reward e policy si modificano alternativamente e possono entrare in un
   regime oscillatorio.

### 13.1 Interpretazione corretta delle metriche

#### AUC alta

`state_action_auc > 0.9` non indica una buona imitazione. La metrica allena un
classificatore lineare a distinguere transizioni esperte e transizioni agente
usando osservazioni e azioni. Un valore alto significa che le due occupancy
sono ancora molto differenti. Se l'agente imitasse l'esperto, l'AUC dovrebbe
avvicinarsi a 0.5.

La metrica non usa `next_status` e `done`, quindi un valore sopra 0.9 indica una
differenza già presente negli stati e nelle azioni, non soltanto negli esiti
terminali.

#### Return standard deviation e margine

Le diagnostiche calcolano return grezzi sommati sull'intera traiettoria. Valori
di `return_std` e margine molto alti mostrano che la scala o la dispersione dei
return sta crescendo, ma non dimostrano che il reward rappresenti correttamente
il task.

Queste metriche sono inoltre confuse dalla lunghezza:

$$
R_\theta(\tau)=\sum_{t=0}^{T-1}r_\theta(s_t,a_t,s_{t+1}).
$$

Due traiettorie con reward per-step simili possono avere return molto diversi
soltanto perché hanno durate differenti. Per questo è utile affiancare alle
metriche esistenti reward medio per step e statistiche condizionate per stato
terminale.

#### ESS fraction molto bassa

Per `maxent_corrected`, i logits e i pesi sono:

$$
z_i=\frac{R_\theta(\tau_i)}{\beta}-\log q(\tau_i),
\qquad
w_i=\frac{e^{z_i}}{\sum_j e^{z_j}}.
$$

L'ESS vale:

$$
ESS=\frac{1}{\sum_iw_i^2}.
$$

Una effective sample fraction sotto 0.01 indica che solo una piccola parte del
batch contribuisce realmente alla partition function. Per interpretarla serve
però controllare anche:

- ESS assoluta;
- peso top-1;
- massa top-5;
- numero totale di elementi nella partition.

Con molti frammenti, una fraction piccola non implica necessariamente
`ESS = 1`. Se però il peso top-1 è vicino a uno, il collasso è inequivocabile.

#### Norma del gradiente intermittente

Il gradiente della loss può essere scritto schematicamente come:

$$
\nabla_\theta\mathcal L
=-
\mathbb E_{\tau_e}
[\nabla_\theta R_\theta(\tau_e)]
+
\sum_iw_i\nabla_\theta R_\theta(\tau_i).
$$

Quando i pesi sono quasi one-hot:

$$
\nabla_\theta\mathcal L
\approx
-\mathbb E_{\tau_e}[\nabla_\theta R_\theta(\tau_e)]
+\nabla_\theta R_\theta(\tau_{top}).
$$

Il training dipende quindi dalla singola traiettoria dominante del batch. Il
cambio della traiettoria top può produrre gradienti grandi; cancellazioni fra i
due termini, saturazione delle unità nascoste o batch già separati possono
produrre gradienti quasi nulli. L'alternanza 80--0 è perciò coerente con una
partition degenerata, anche se da sola non ne dimostra la causalità.

### 13.2 Perché l'importance sampling collassa

Su una traiettoria intera vengono sommati sia reward sia log-density:

$$
R(\tau)=\sum_t r_t,
\qquad
\log q(\tau)=\sum_t\log\pi(a_t\mid s_t).
$$

Anche differenze moderate per step diventano grandi dopo molti step. Poiché i
pesi dipendono dall'esponenziale dei corrected logits, una differenza di poche
decine è sufficiente a rendere la softmax quasi one-hot.

L'AUC sopra 0.9 fornisce un secondo indizio: la proposal SAC visita una
distribuzione molto diversa da quella esperta. L'importance sampling è
formalmente possibile se la proposal ha supporto sufficiente, ma diventa
statisticamente inefficiente quando la proposal assegna probabilità pratica
molto bassa alle regioni importanti.

[Guided Cost Learning](https://proceedings.mlr.press/v48/finn16.html) usa una
proposal adattiva e importance weights proprio per avvicinare progressivamente
la distribuzione di campionamento a quella definita dal reward. La formula nel
codice è coerente con questa idea, ma l'alternanza con SAC non garantisce da
sola che la proposal raggiunga rapidamente la distribuzione target.

**Conclusione:** l'ESS molto bassa è principalmente un problema statistico e
algoritmico dell'importance sampling a orizzonte lungo. Non è necessario che
esista un errore algebrico in `losses.py` perché si verifichi.

### 13.3 Perché l'off-road breve può diventare ottimo

Il reward model usa esplicitamente:

$$
(s_t,a_t,\texttt{next\_status}_t,\texttt{done}_t).
$$

Il termine esperto aumenta il reward delle transizioni esperte. La
penalizzazione delle transizioni off-road, assenti o rare nell'esperto, deve
arrivare soprattutto dalla partition sui rollout dell'agente.

Se la partition è dominata da pochissime traiettorie, la maggior parte degli
episodi off-road riceve peso quasi nullo. Il reward può quindi separare molto
bene esperto e agente in media, producendo AUC e margine alti, senza assegnare
una penalità sufficiente a tutte le modalità di fallimento.

SAC può preferire la terminazione se:

$$
r_{offroad}
>
r_{running}+\gamma V(s').
$$

Questo può accadere anche quando $r_{offroad}<0$: terminare evita una lunga
sequenza di reward running ancora più negativi.

L'ambiente rende questa scorciatoia facile da raggiungere. Una richiesta di
cambio corsia verso una corsia inesistente imposta `off_road`, e ogni stato
diverso da `running` termina l'episodio. Non è necessariamente un bug
dell'ambiente, ma è una superficie di reward hacking molto accessibile.

### 13.4 Normalizzazione agent-facing e durata dell'episodio

Se nella run sul server è attiva la normalizzazione del reward consegnato a SAC,
la trasformazione è:

$$
r'_t=\frac{r_t-\mu}{\sigma}.
$$

La divisione per una costante positiva modifica la scala. La sottrazione di una
costante per step non è invece policy-invariant quando la durata è controllabile:

$$
R'(\tau)
=\frac{R(\tau)-|\tau|\mu}{\sigma}.
$$

Il termine $-|\tau|\mu$ può introdurre direttamente una preferenza per episodi
brevi oppure lunghi. Inoltre media e deviazione standard vengono aggiornate
usando il rollout corrente, rendendo non stazionario anche il reward percepito
dall'agente.

Questa è una causa condizionale: va verificata sulla configurazione effettiva
del server, non su quella locale.

### 13.5 Mismatch fra obiettivo MaxEnt e obiettivo SAC

La loss costruisce return non scontati:

$$
R(\tau)=\sum_t r_t.
$$

SAC ottimizza normalmente un obiettivo scontato ed entropy-regularized:

$$
J_{SAC}
=\mathbb E
\left[
\sum_t\gamma^t
\left(r_t+\alpha\mathcal H(\pi(\cdot\mid s_t))\right)
\right].
$$

La temperatura $\beta$ della likelihood IRL, il coefficiente entropico
$\alpha$ di SAC e il discount $\gamma$ non vengono resi coerenti
automaticamente dall'implementazione.

Di conseguenza SAC non è necessariamente una procedura di policy optimization
per la stessa distribuzione:

$$
q(\tau)\propto\exp(R(\tau)/\beta)
$$

usata nella derivazione della loss. Questo può impedire l'appiattimento dei
pesi IS anche quando la formula della partition è implementata correttamente.
Si tratta di un problema d'integrazione fra i due algoritmi, non di un semplice
bug locale. L'obiettivo originale di SAC è descritto in
[Soft Actor-Critic](https://proceedings.mlr.press/v80/haarnoja18b.html).

### 13.6 Perché i frammenti non risolvono necessariamente

I frammenti accorciano la somma di `log q`, ma non eliminano:

- la distanza fra proposal ed esperto;
- la possibile crescita della scala del reward;
- il mismatch fra SAC e distribuzione MaxEnt;
- la non stazionarietà del reward;
- la scorciatoia di terminazione.

Introducono anche un peso implicito legato alla durata. Con frammenti di
lunghezza $K$, una traiettoria di lunghezza $T$ produce circa $T/K$ elementi.
Una traiettoria riuscita e lunga può quindi contribuire molte volte alla
partition, mentre un off-road breve contribuisce una o poche volte. L'obiettivo
locale può così penalizzare numericamente più finestre delle traiettorie lunghe
rispetto ai fallimenti brevi.

Inoltre i frammenti della stessa traiettoria sono correlati. L'ESS calcolata
trattandoli come elementi distinti non equivale all'ESS di altrettanti campioni
i.i.d. Le oscillazioni osservate con $K=10$ o $K=30$ sono quindi compatibili con
un miglioramento insufficiente della varianza e con un obiettivo locale diverso
da quello a traiettoria intera.

### 13.7 Perché il success rate può oscillare

Il ciclo di training alterna due problemi non stazionari:

1. il reward viene aggiornato per separare esperto e policy corrente;
2. SAC ottimizza il nuovo reward;
3. la nuova policy genera hard negatives diversi;
4. il reward cambia nuovamente;
5. critic e replay contengono informazione prodotta sotto reward precedenti.

Con reward memorizzati nel replay, SAC usa target stale. Con relabeling, i reward
dei campioni possono essere ricalcolati, ma critic, target network e
distribuzione del replay rimangono il risultato delle iterazioni precedenti.
Il relabeling riduce quindi il problema, ma non rende stazionario il gioco.

Questo meccanismo può produrre cicli nei quali l'agente trova un comportamento
riuscito, il reward si adatta, il comportamento perde valore relativo e la
policy torna verso collisioni o off-road.

### 13.8 Valutazione delle possibili origini

| Componente | Valutazione |
|---|---|
| Formula di `maxent_corrected` | non emerge un errore algebrico evidente |
| Importance sampling a traiettoria intera | causa molto probabile del collasso ESS |
| Integrazione MaxEnt--SAC | probabile sorgente di mancata convergenza |
| Reward terminale/duration bias | causa molto probabile della scorciatoia off-road |
| Normalizzazione con sottrazione della media | causa seria se attiva sul server |
| Replay e reward non stazionario | probabile amplificatore delle oscillazioni |
| Ambiente SUMO | non necessariamente errato, ma rende facile la terminazione off-road |
| SAC isolato | possibile amplificatore, non spiegazione sufficiente da solo |

Rimane da verificare numericamente che le log-density salvate per SAC
corrispondano esattamente alla densità dell'azione campionata, inclusi
trasformazione `tanh` e scala dell'action space. I test esistenti verificano la
formula generale e diversi componenti, ma non costituiscono una validazione
end-to-end completa delle log-density SAC nelle run reali.

### 13.9 Esperimenti diagnostici discriminanti

#### 1. Congelare il reward model

Congelare un checkpoint del reward e continuare ad allenare SAC:

- se le oscillazioni spariscono, la causa principale è il loop reward--policy;
- se off-road e oscillazioni restano, il problema è già presente nel reward
  fisso, nella trasformazione agent-facing, in SAC o nell'ambiente.

#### 2. Confrontare i return per esito

Su un reward congelato, valutare separatamente:

- traiettorie esperte;
- successi dell'agente;
- off-road brevi;
- collisioni;
- policy casuale.

Per ogni gruppo calcolare:

- return raw non scontato;
- reward medio per step;
- return dopo la trasformazione agent-facing;
- return scontato effettivamente rilevante per SAC;
- lunghezza e reward terminale.

Se gli off-road superano i successi secondo il return agent-facing scontato, la
scorciatoia è dimostrata direttamente.

#### 3. Scomporre i corrected logits

Per ogni traiettoria loggare:

$$
T,\quad
\text{stato terminale},\quad
R/\beta,\quad
-\log q,\quad
z=R/\beta-\log q,\quad
w=\operatorname{softmax}(z).
$$

Poi misurare correlazione di ciascun termine con lunghezza e stato terminale.
Lo stato delle top-5 traiettorie per peso rivela quali esempi stanno realmente
allenando la partition.

#### 4. Verificare la densità SAC

Sul server, confrontare per le stesse osservazioni e azioni:

- log-density restituita direttamente dalla distribuzione dell'attore SAC;
- `log_policy_prob` registrata nella transizione;
- log-density ricalcolata da `policy_action_log_probs`.

La verifica deve includere azioni interne e azioni vicine ai limiti.

#### 5. Baseline SAC con reward vero

Allenare SAC nello stesso ambiente con il reward vero:

- se riesce, agente e ambiente sono capaci di risolvere il task e il problema è
  nel reward learning o nella sua integrazione;
- se fallisce allo stesso modo, vanno analizzati controller continuo, azioni di
  cambio corsia e dinamica di terminazione.

### 13.10 Conclusione e confidenza

La diagnosi più probabile è:

1. **molto probabile:** collasso dell'importance sampling per proposal distante
   e orizzonte lungo;
2. **molto probabile:** reward hacking tramite terminazione off-road;
3. **probabile:** instabilità del loop alternato reward--SAC;
4. **probabile se attiva:** normalizzazione con mean subtraction;
5. **possibile:** mismatch fra obiettivo SAC e distribuzione MaxEnt;
6. **meno probabile:** errore puro nella formula di `losses.py`;
7. **da verificare:** correttezza numerica end-to-end delle log-density SAC.

**Confidenza complessiva: 85%.**

La confidenza è alta sul collasso dell'importance sampling e sul meccanismo che
può rendere conveniente una terminazione precoce. È moderata sull'origine
esatta delle oscillazioni: l'esperimento con reward congelato è necessario per
separare il gioco reward--policy dall'instabilità di SAC sotto un reward fisso.
