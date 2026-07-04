# Refactoring di `DemoAlgorithm`

## Obiettivo

Il file `human_feedback_rl/algorithms/demo_algorithm.py` conteneva circa 930
righe e riuniva responsabilità differenti: orchestrazione del training, loss
IRL, ottimizzazione del reward model, rollout, diagnostica e checkpoint.

Il refactoring separa queste responsabilità in moduli dedicati, mantenendo
`DemoAlgorithm` come punto di ingresso pubblico e preservando il comportamento
esistente.

## Struttura risultante

```text
human_feedback_rl/algorithms/
├── __init__.py
├── demo_algorithm.py
└── demo/
    ├── __init__.py
    ├── losses.py
    ├── reward_training.py
    ├── rollout.py
    ├── reward_diagnostics.py
    ├── imitation_metrics.py
    └── checkpointing.py
```

## Responsabilità dei file

### `demo_algorithm.py`

È la facciata pubblica dell'algoritmo e contiene:

- la classe `DemoAlgorithm`;
- la validazione dei parametri di inizializzazione;
- la creazione del reward model, degli optimizer e del trajectory generator;
- il metodo `train`, che orchestra il ciclo alternato;
- il logging conclusivo di ogni iterazione;
- le costanti pubbliche relative a loss, status e classificatore.

Il flusso principale rimane:

```text
raccolta rollout
    → diagnostica pre-update
    → training del reward model
    → normalizzazione del reward per l'agente
    → diagnostica post-update
    → training dell'agente
    → checkpoint
```

### `demo/losses.py`

Contiene `RewardLossMixin`, responsabile di:

- campionamento delle traiettorie expert e model;
- calcolo dei ritorni differenziabili;
- implementazione delle loss `maxent`, `maxent_2`, `demo`, `demo_loss`,
  `maxent_corrected` e `demo_corrected`;
- calcolo della probabilità della traiettoria sotto la policy;
- costante `VALID_LOSSES`.

Le formule delle loss storiche sono rimaste invariate.

### `demo/reward_training.py`

Contiene `RewardTrainingMixin`, responsabile di:

- training dei membri del reward ensemble;
- backward pass e aggiornamento degli optimizer;
- controllo di loss e gradienti non finiti;
- calcolo delle norme di gradienti e parametri;
- aggiornamento della normalizzazione del reward usato dall'agente.

### `demo/rollout.py`

Contiene `RolloutMixin`, responsabile di:

- raccolta delle traiettorie dall'agente;
- logging delle statistiche dei rollout;
- controllo delle azioni sui limiti dello spazio continuo;
- valutazione delle traiettorie con il reward model;
- training dell'agente.

### `demo/reward_diagnostics.py`

Contiene `RewardDiagnosticsMixin`, responsabile di:

- diagnostica specifica delle loss;
- effective sample size delle distribuzioni MaxEnt;
- validazione del reward prima e dopo gli aggiornamenti;
- ensemble disagreement;
- ranking dei ritorni mediante correlazione di Spearman;
- confronto tra reward memorizzati e reward ricalcolati nel replay buffer;
- conversione di una sequenza di transizioni in traiettorie.

Queste operazioni osservano il training, ma non modificano direttamente la loss
o la policy.

### `demo/imitation_metrics.py`

Contiene `ImitationMetricsMixin`, responsabile di:

- estrazione delle feature stato-azione;
- divisione train/validation a livello di traiettoria;
- bilanciamento dei campioni expert e agent;
- classificatore logistico diagnostico;
- calcolo dell'AUC.

### `demo/checkpointing.py`

Contiene `CheckpointingMixin`, responsabile del salvataggio di:

- reward model;
- configurazione e stato degli optimizer;
- agente;
- replay buffer, quando supportato.

Il formato e i nomi dei file di checkpoint sono rimasti invariati.

## Uso dei mixin

`DemoAlgorithm` usa l'ereditarietà multipla:

```python
class DemoAlgorithm(
    RewardLossMixin,
    RewardTrainingMixin,
    RolloutMixin,
    ImitationMetricsMixin,
    RewardDiagnosticsMixin,
    CheckpointingMixin,
    BaseAlgorithm,
):
    ...
```

`DemoAlgorithm` è la classe figlia. I mixin e `BaseAlgorithm` sono classi base.
I mixin non rappresentano oggetti separati: aggiungono metodi alla stessa
istanza di `DemoAlgorithm`.

Per esempio:

```python
algorithm = DemoAlgorithm(...)
algorithm._update_agent_reward_normalization()
```

Python trova `_update_agent_reward_normalization` in `RewardTrainingMixin`, ma
il valore di `self` continua a essere `algorithm`. Di conseguenza il metodo può
leggere attributi inizializzati da `DemoAlgorithm.__init__`, come:

```python
self.normalize_agent_reward
self.trajectories
self.reward_model
self.logger
```

Il contratto implicito di ogni mixin è quindi che `DemoAlgorithm` inizializzi lo
stato richiesto dai suoi metodi.

## Compatibilità

L'import pubblico non è cambiato:

```python
from human_feedback_rl.algorithms import DemoAlgorithm
```

Sono rimasti invariati anche:

- firma del costruttore e del metodo `train`;
- nomi dei metodi usati dai test;
- costanti accessibili tramite `DemoAlgorithm`;
- chiavi di logging;
- formato dei checkpoint;
- formule delle loss;
- integrazione con replay buffer e Stable-Baselines3.

## Verifica

Dopo il refactoring sono stati eseguiti i test disponibili del package mediante
l'ambiente `sumo-rlhf`:

```text
Ran 12 tests
OK
```

Sono stati inoltre verificati:

- compilazione dei moduli Python;
- assenza di errori tramite `git diff --check`;
- funzionamento del salvataggio dei checkpoint;
- accessibilità dei precedenti metodi privati attraverso `DemoAlgorithm`.
