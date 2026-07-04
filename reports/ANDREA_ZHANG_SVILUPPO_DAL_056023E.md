# Attività svolte da Zhang Andrea dal commit `056023e`

## Perimetro e criterio di lettura

Questa relazione ricostruisce il lavoro attribuito in Git ad **Andrea Zhang
<andrea02polimi@gmail.com>** dal commit
`056023e7f8428e66ce59bd0d6239869230579c47` della repository principale
`sumo-human-feedback-rl` fino allo stato locale del **21 giugno 2026**, comprese
le modifiche non ancora commitate.

Sono state considerate due cronologie:

- la repository principale `sumo-human-feedback-rl`, dal commit indicato a
  `3dd9acb` incluso;
- il submodule `human-feedback-rl`, dal commit registrato dalla repository
  principale (`df135068`) al checkout locale `24a1727d`, oltre al refactoring
  non commitato.

I paragrafi **Cosa è stato fatto** derivano dai diff Git. I paragrafi **Perché**
ricostruiscono lo scopo dai messaggi di commit, dai commenti nel codice, dai
test e dal rapporto tra modifiche coordinate nelle due repository. Quando Git
non permette di provare l'intenzione dell'autore, la motivazione va quindi
letta come motivazione tecnica inferita.

## Sintesi del lavoro

Nel periodo analizzato il lavoro si è concentrato su cinque obiettivi:

1. costruire una procedura riproducibile di tuning della baseline SAC;
2. ripulire e uniformare gli entry point degli esperimenti;
3. correggere e rendere più robusto il ciclo Demo IRL, soprattutto con SAC;
4. rendere osservabili reward learning, replay buffer e qualità
   dell'imitazione tramite W&B;
5. separare il grande `DemoAlgorithm` in moduli con responsabilità più chiare.

Il risultato non è soltanto l'aggiunta di nuove funzionalità. Una parte
significativa del lavoro riguarda la validità sperimentale: valutazioni su più
seed, isolamento dell'ambiente di rollout, distinzione fra loss storiche e
corrette, controllo della non stazionarietà del reward nel replay buffer e
diagnostiche che non influenzano direttamente il training.

## Commit della repository principale `sumo-human-feedback-rl`

### 18 giugno 2026 — `056023e` — `SAC tuning`

**Cosa è stato fatto.** È stata introdotta una pipeline completa per il tuning
della baseline SAC sul reward reale dell'ambiente:

- nuova configurazione Hydra `configs/train_sac_baseline.yaml`;
- nuovo entry point `scripts/train_sac_baseline.py` per addestramento,
  valutazione deterministica e aggregazione dei risultati;
- tre sweep W&B progressivi: parametri fondamentali di SAC, replay
  buffer/schedule degli update e architettura della rete;
- launcher parallelo `sweeps/run_agents.sh`, con un agente W&B per core e limite
  a un thread matematico per processo;
- valutazione su episodi e seed distinti da quelli di training, logging di
  return, velocità ed esiti terminali, salvataggio degli agenti per seed.

**Perché.** Lo scopo era ottenere una baseline SAC ben calibrata e un confronto
attendibile con gli algoritmi di imitation/reward learning. La suddivisione del
tuning in stadi riduce lo spazio di ricerca; l'aggregazione su più seed evita di
selezionare una configurazione fortunata; il parallelismo per core sfrutta il
fatto che il collo di bottiglia è la simulazione SUMO, prevalentemente
single-thread.

### 18 giugno 2026 — `e5f4347` — `fix for sac tuning`

**Cosa è stato fatto.** Sono stati corretti problemi operativi della prima
versione del tuning:

- sostituito il singolo seed con `[0, 1, 2]`;
- assegnato `outputs` come directory predefinita reale, evitando il crash
  causato da `Null` in `make_run_dir`;
- generati nomi W&B informativi a partire dagli override Hydra del trial;
- chiarito nel launcher che un'uscita immediata degli agenti può indicare un
  crash e che vanno controllati i log.

**Perché.** Il tuning doveva poter partire senza override obbligatori, produrre
risultati robusti ai seed ed essere diagnosticabile rapidamente quando molti
worker paralleli falliscono all'avvio.

### 18 giugno 2026 — `a37f578` — `script for extracting best parameters from sweep`

**Cosa è stato fatto.** È stato aggiunto `sweeps/best_params.py`, che interroga
la API W&B, considera solo run terminate con la metrica richiesta, le ordina
secondo l'obiettivo dello sweep, mostra le migliori con media e deviazione
standard e stampa gli `agent.kwargs` vincenti in YAML.

**Perché.** La pipeline di tuning è sequenziale: i migliori parametri di uno
stadio devono diventare i valori fissi dello stadio successivo. Automatizzare
l'estrazione riduce errori di trascrizione e rende esplicito il criterio con cui
viene scelta la configurazione migliore.

### 19 giugno 2026 — `4fb654f` — `code clean up and new debugging strategy`

**Cosa è stato fatto.** Gli script CHRI/PPO, Demo/PPO, Demo/SAC e SAC baseline
sono stati semplificati estraendo in `scripts/_common.py` le operazioni comuni:
creazione delle directory, seeding di Python/NumPy/PyTorch, caricamento dei
dataset e inizializzazione W&B. Sono stati inoltre:

- uniformati i percorsi delle configurazioni Hydra;
- reso opzionale il dataset di debug;
- eliminato codice duplicato e import inutilizzati;
- corretto il parametro `alfa` in `alpha` nelle configurazioni;
- esposto `loss_type` per Demo/SAC e aggiunto `log_interval` agli esperimenti.

**Perché.** Centralizzare il boilerplate evita divergenze silenziose fra gli
esperimenti e rende ogni entry point più facile da verificare. I nuovi parametri
servono inoltre a cambiare loss e frequenza di logging senza modificare il
codice, cioè a sostenere una strategia di debug basata su configurazioni
ripetibili.

### 19 giugno 2026 — `562e3ab` — `feat: add configurable PPO and SAC demo launchers`

**Cosa è stato fatto.** Sono stati creati launcher shell espliciti per Demo PPO
e Demo SAC e un launcher di ablation appaiata per confrontare il relabeling dei
reward sugli stessi seed. Le configurazioni distinguono loss storiche
(`maxent`, `maxent_2`, `demo`) e corrette (`maxent_corrected`,
`demo_corrected`). Gli script Python ora:

- includono loss e relabeling nel nome della run;
- creano un ambiente separato per i rollout, con seed distinto;
- usano per SAC il nuovo `RewardRelabelReplayBuffer`;
- inoltrano `rollout_env` e `relabel_rewards` a `DemoAlgorithm`.

**Perché.** I launcher rendono visibili e modificabili tutti gli iperparametri
importanti, facilitando esperimenti riproducibili. L'ambiente separato impedisce
che la raccolta delle traiettorie desincronizzi l'ambiente usato da SB3. Il test
appaiato del relabeling isola il suo effetto dalla variabilità dovuta al seed.

### 19 giugno 2026 — `91d39a5` — `feat(logging): configure semantic W&B metrics for training runs`

**Cosa è stato fatto.** L'inizializzazione W&B comune chiama
`configure_wandb_metrics`, definita nel submodule, per associare famiglie di
metriche agli assi temporali corretti.

**Perché.** Le metriche dell'agente avanzano in timestep, mentre reward model e
diagnostiche avanzano per iterazione esterna. Usare assi semantici evita grafici
fuorvianti e rende confrontabili curve prodotte con frequenze di logging
diverse.

### 20 giugno 2026 — `3b8f066` — `agent reward normalization`

**Cosa è stato fatto.** È stata esposta nelle configurazioni e nei launcher
l'opzione `normalize_agent_reward`. Nel launcher SAC sono stati anche aggiornati
i valori sperimentali della loss, del learning rate del reward, dei gradient
step, delle dimensioni dei batch/reti e dell'intervallo di iterazione.

**Perché.** La normalizzazione porta il reward consumato dalla policy a una
scala più stabile senza cambiare il reward grezzo usato dalla loss IRL. Questo
riduce la sensibilità di PPO/SAC al drift di scala del reward model. Gli altri
valori configurano una nuova prova coerente con questa modalità.

### 20 giugno 2026 — `119c453` — `hidden some plots on wandb`

**Cosa è stato fatto.** La directory di output del launcher SAC è stata
spostata da `outputs/demo_sac` a `outputs/demo_sac_debug`.

**Perché.** Il cambiamento separa le run di debug dagli esperimenti ordinari e
va letto insieme al commit omonimo del submodule, che riduce i pannelli W&B
visibili. Il messaggio del commit descrive soprattutto quella modifica
coordinata, mentre nella repository principale cambia soltanto la destinazione
degli artefatti.

### 21 giugno 2026 — `3dd9acb` — `feat: add state-action imitation AUC diagnostics to W&B`

**Cosa è stato fatto.** Nelle configurazioni Demo PPO e SAC è stato aggiunto
`imitation_diagnostics_interval`, con valore `10` e possibilità di disabilitare
la diagnostica impostandolo a zero.

**Perché.** Il calcolo dell'AUC richiede di addestrare un classificatore
diagnostico e non è gratuito. Renderne configurabile la frequenza permette di
misurare periodicamente la somiglianza fra occupancy expert e agente senza
pagare il costo a ogni iterazione.

## Commit del submodule `human-feedback-rl`

### Nota sui tre commit del 17 giugno

I commit `82bf44d`, `09261b3` e `4dcb5d8` sono cronologicamente precedenti al
commit iniziale richiesto del 18 giugno. Sono inclusi perché si trovano comunque
nel delta fra il commit del submodule registrato dalla repository principale
(`df135068`) e il checkout locale corrente. Ometterli renderebbe incompleta la
storia effettivamente presente nel submodule locale.

### 17 giugno 2026 — `82bf44d` — `added log interval to learn`

**Cosa è stato fatto.** È stato inoltrato l'intervallo di logging al training
dell'agente e aggiunta una seconda formulazione sperimentale della loss MaxEnt,
poi denominata `maxent_2`.

**Perché.** La frequenza di logging deve essere controllabile dall'esperimento,
specialmente quando PPO e SAC producono episodi e update a ritmi diversi. La
seconda loss permette di confrontare diverse approssimazioni del termine di
partizione MaxEnt.

### 17 giugno 2026 — `09261b3` — `log gradient norm loss reward model`

**Cosa è stato fatto.** È stato calcolato e registrato il valore medio della
norma L2 complessiva dei gradienti del reward ensemble durante il training.

**Perché.** La norma dei gradienti permette di individuare gradienti nulli,
instabili o esplosivi e di distinguere un problema di ottimizzazione del reward
model da un problema della policy.

### 17 giugno 2026 — `4dcb5d` — `spearman on trajectories returns and ensemble std`

**Cosa è stato fatto.** Il dataset piatto di debug viene ricostruito in
traiettorie. Sono state aggiunte:

- correlazione di Spearman fra return reale e return predetto;
- deviazione standard fra membri dell'ensemble, globale e sugli stati running;
- valutazione sia sul rollout corrente sia sul dataset di debug fisso.

**Perché.** Nell'IRL è importante che il reward ordini correttamente le
traiettorie, non soltanto che predica bene ogni transizione. Spearman misura
questa qualità di ranking; il disaccordo dell'ensemble segnala invece esempi
fuori distribuzione o zone in cui il reward è incerto.

### 19 giugno 2026 — `fcf7943` — `code clean up and new debugging strategy`

**Cosa è stato fatto.** `DemoAlgorithm` e i wrapper sono stati riorganizzati e
documentati. `loss_type` è diventato un parametro validato; le diverse loss sono
state instradate da un unico metodo; sono state aggiunte diagnostiche specifiche
per loss e return, statistiche per status terminale, norme dei parametri,
effective sample size e un primo controllo della differenza fra reward salvati
e reward correnti. `EnvRewardWrapper` conserva un campione di transizioni per
questa analisi e `NormalizedRewardNet` è stato semplificato.

**Perché.** L'obiettivo era passare da un debug generico a uno capace di spiegare
*perché* il reward model fallisce: saturazione del softmax, scala incontrollata,
scorciatoie basate sul terminal status, incertezza dell'ensemble oppure reward
obsoleti nel replay di SAC.

### 19 giugno 2026 — `33a15ca` — `feat: harden demo IRL training and SAC reward relabeling`

**Cosa è stato fatto.** Questo è il principale intervento di robustezza
algoritmica:

- conservate le loss storiche e aggiunte `maxent_corrected` e
  `demo_corrected`;
- per MaxEnt corretta, usata la log-probabilità della policy proposta e richiesto
  un `rollout_env` dedicato;
- corretto il calcolo della densità per azioni continue/clippate di PPO e SAC;
- introdotto il bootstrap del reward model prima che l'agente apprenda da un
  reward casuale;
- introdotto `RewardRelabelReplayBuffer`, capace di ricalcolare i reward con il
  modello corrente oppure conservare quelli memorizzati per l'ablation;
- isolata la raccolta dei rollout dall'ambiente di training e gestiti
  correttamente rollout multi-env;
- aggiunti controlli su loss/gradienti non finiti, checkpoint più completi e
  286 righe di test dedicati.

**Perché.** SAC è off-policy: se il reward model cambia, il replay buffer
contiene target prodotti da versioni diverse del reward. Il relabeling affronta
questa non stazionarietà; il bootstrap evita di riempire subito il buffer con un
reward casuale. La correzione per la proposal policy rende coerente la stima
MaxEnt quando le traiettorie non provengono da una distribuzione fissa implicita.
L'ambiente separato previene inoltre la desincronizzazione dello stato interno
di SB3/SUMO.

### 19 giugno 2026 — `fdc2676` — `feat(logging): improve W&B axes and reward diagnostics`

**Cosa è stato fatto.** È stata aggiunta `configure_wandb_metrics` con assi
distinti per timestep dell'agente e iterazioni IRL, insieme a regole per
nascondere metriche secondarie. La validazione del reward ora registra anche i
gap fra reward medi degli status, per esempio arrivo rispetto a collisione.

**Perché.** Gli assi corretti impediscono di confrontare punti appartenenti a
scale temporali diverse. I gap tra status aiutano a scoprire se il modello ha
imparato una separazione utile o una scorciatoia legata esclusivamente al flag
terminale.

### 19 giugno 2026 — `4a3c900` — `fix(logging): prevent SB3 metric name collisions`

**Cosa è stato fatto.** È stato creato un formato testuale SB3 più largo e
utilizzato anche da `BaseAlgorithm` e DAgger. Sono stati aggiunti test con chiavi
lunghe e simili, come `ensemble_std` ed `ensemble_std_running`.

**Perché.** Il formato umano di SB3 tronca i nomi lunghi; metriche diagnostiche
annidate potevano quindi collidere, nascondersi a vicenda o causare errori. Una
larghezza adeguata preserva chiavi univoche senza rinunciare al logging su
console.

### 20 giugno 2026 — `3d73467` — `agent reward normalization`

**Cosa è stato fatto.** `DemoAlgorithm` può ora stimare media e deviazione
standard del reward grezzo sul rollout corrente e applicare la trasformazione
soltanto al reward consumato dall'agente e dal relabeling. La loss IRL continua
a usare il forward grezzo. Sono stati aggiunti controlli di finitezza, gestione
del caso a varianza nulla, logging delle statistiche e test di non interferenza
con il training del reward model.

**Perché.** Separare le due scale stabilizza i target di PPO/SAC senza alterare
l'obiettivo IRL. In questo modo si evita che un semplice drift di scala del
reward ensemble cambi drasticamente la dinamica di apprendimento della policy.

### 20 giugno 2026 — `cda9207` — `hidden some plots on wandb`

**Cosa è stato fatto.** La configurazione W&B è passata da molte wildcard
visibili a una lista esplicita di metriche essenziali. Le metriche secondarie
vengono definite dinamicamente come esatte e nascoste, mantenendo assi corretti.
Sono stati aggiunti test per il comportamento specifico di W&B 0.27.

**Perché.** Il numero elevato di metriche diagnostiche generava automaticamente
troppi pannelli. Nascondere per default le serie secondarie rende la dashboard
leggibile senza perdere i dati, che restano disponibili per analisi mirate.

### 21 giugno 2026 — `24a1727` — `feat: add state-action imitation AUC diagnostics to W&B`

**Cosa è stato fatto.** È stata aggiunta una diagnostica di imitazione che:

- estrae feature congiunte stato-azione da traiettorie expert e agent;
- divide train e validation a livello di traiettoria, evitando leakage;
- bilancia le due classi e standardizza usando soltanto il training set;
- addestra un classificatore logistico nuovo a ogni misura;
- registra l'AUC held-out e il tempo di calcolo in W&B.

**Perché.** Il return ambientale da solo non dice se la policy occupa regioni
state-action simili a quelle dell'expert. L'AUC funziona come two-sample test:
un valore vicino a `0.5` indica distribuzioni difficili da distinguere, mentre
un valore alto segnala che l'agente è ancora separabile dall'expert. La metrica è
solo osservativa e non retroagisce su reward o policy.

## Stato locale non commitato al 21 giugno 2026

Le modifiche seguenti sono presenti nel workspace ma non hanno ancora autore,
data o motivazione certificabili tramite Git. Sono incluse perché costituiscono
lo stato più recente richiesto.

### Refactoring di `DemoAlgorithm` nel submodule

**Cosa è stato fatto.** Il file `demo_algorithm.py`, arrivato a circa 930 righe,
è stato ridotto alla facciata e all'orchestrazione del ciclo di training. La
logica è stata distribuita in mixin sotto
`human_feedback_rl/algorithms/demo/`:

- `losses.py`: formule IRL e probabilità delle traiettorie;
- `reward_training.py`: ottimizzazione, gradienti e normalizzazione;
- `rollout.py`: raccolta delle traiettorie e training dell'agente;
- `reward_diagnostics.py`: validazione, ESS, Spearman e replay staleness;
- `imitation_metrics.py`: classificatore state-action e AUC;
- `checkpointing.py`: persistenza di modello, optimizer, agente e replay.

La classe pubblica `DemoAlgorithm` compone questi mixin mantenendo invariati
import, firma, metodi usati dai test, chiavi di logging, loss e formato dei
checkpoint. Sono stati aggiunti test sintetici per verificare che l'AUC distingua
distribuzioni separate e resti circa casuale per distribuzioni uguali. Il file
`DEMO_ALGORITHM_REFACTORING.md` documenta struttura, contratti impliciti dei
mixin e compatibilità; riporta 12 test superati.

**Perché.** La classe monolitica mescolava orchestrazione, matematica delle loss,
I/O, rollout e diagnostica. La separazione riduce il carico cognitivo e il
rischio di conflitti durante modifiche future, pur conservando una singola API
pubblica e lo stesso comportamento sperimentale.

### Nuova configurazione locale di debug SAC

**Cosa è stato fatto.** `launchers/run_demo_SAC.sh` punta al progetto W&B
`demo-sac-debug` e prova una nuova configurazione: `maxent_corrected`, reward
relabeling attivo, normalizzazione agente disattiva, `gamma=0.995`, `tau=0.005`,
16 gradient step SAC, learning rate reward `3e-4`, 20 update reward, L2 `0.05` e
diagnostica AUC ogni 5 iterazioni.

**Perché.** La configurazione isola una run di debug e combina la loss corretta
con relabeling esplicito. La maggiore regolarizzazione e il minor numero di
update del reward sembrano mirare a verificare se i problemi osservati derivano
da saturazione/overfitting del reward model; l'AUC più frequente rende visibile
più rapidamente l'effetto sulla distribuzione state-action.

### Dataset e analisi sperimentali locali

**Cosa è stato fatto.** Nel workspace sono presenti:

- una versione modificata e molto più grande di
  `data_for_training/expert_trajectories.pkl`;
- dataset non tracciati per valutazione bilanciata e AIRL
  (`balanced_eval_dataset*.pkl`, `expert_trajectories_airl.pkl`);
- notebook per analizzare i dati e rimuovere in modo verificato le traiettorie
  expert con collisione;
- guide alle metriche e ai grafici W&B, analisi della run `63jhx6wt` e un
  notebook di diagnostica automatica;
- esportazione JSON W&B, documentazione degli sweep e `environment.yml` con le
  versioni dell'ambiente;
- pseudocodice LaTeX/PDF dell'algoritmo e relativi file di compilazione.

**Perché.** Questi artefatti supportano la validazione empirica: costruire
dataset con classi/esiti controllati, evitare che collisioni entrino nelle
dimostrazioni expert, rendere riproducibile l'ambiente software e trasformare le
numerose metriche W&B in una procedura di diagnosi interpretabile. I file
`.aux`, `.fls`, `.fdb_latexmk`, `.out`, `.synctex.gz` e `.DS_Store` sono invece
artefatti generati dagli strumenti e non rappresentano funzionalità del
progetto.

## Relazione fra repository principale e submodule

I commit del 19–21 giugno sono spesso coppie coordinate: il submodule implementa
la logica (`RewardRelabelReplayBuffer`, normalizzazione, configurazione W&B,
AUC), mentre la repository principale la abilita tramite script, configurazioni
e launcher.

C'è però una criticità di versione: in tutti i commit analizzati la repository
principale continua a registrare il submodule al commit `df135068`, mentre il
checkout locale è a `24a1727` con ulteriori modifiche non commitate. Di
conseguenza, un clone pulito della sola repository principale **non riprodurrebbe
automaticamente** il codice del submodule descritto qui. Per rendere lo stato
riproducibile occorrerà prima commitare il refactoring nel submodule e poi
commitare nella repository principale l'aggiornamento del relativo puntatore.

