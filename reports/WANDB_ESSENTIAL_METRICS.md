# Metriche W&B essenziali per Demo SAC

Questa guida descrive i nove pannelli principali consigliati per analizzare
le run Demo SAC. L'elenco completo contiene **20 metriche**: alcune metriche
correlate vanno visualizzate insieme nello stesso pannello.

La whitelist automatica corrente contiene 18 metriche. Le due metriche
aggiuntive descritte qui, `agent/rewards/ep_comfort_return` e
`reward/grad_norm`, restano nella history e possono essere aggiunte manualmente
ai pannelli oppure inserite in `VISIBLE_METRICS` nel logger.

Le metriche `agent/*` devono usare come asse X
`agent/time/total_timesteps`; tutte le altre devono usare `iterations`.
L'asse W&B `_step` non e' adatto, perche' mescola i log interni di SAC con
quelli del ciclo esterno di reward learning.

## 1. Esiti degli episodi

Metriche:

- `agent/event_rate/successes`
- `agent/event_rate/collisions`
- `agent/event_rate/off_road`

### Cosa rappresentano

Sono le frazioni degli episodi recenti terminati rispettivamente con arrivo,
collisione e uscita di strada. Misurano direttamente il comportamento reale
della policy e non dipendono dalla scala del reward appreso.

### Come interpretarle

- `successes` dovrebbe crescere e stabilizzarsi verso 1.
- `collisions` e `off_road` dovrebbero scendere verso 0.
- Un aumento dei successi e delle collisioni contemporaneamente indica una
  policy piu' aggressiva, non un miglioramento privo di compromessi.
- Le singole finestre possono essere rumorose: confrontare medie su piu'
  punti, non soltanto l'ultimo valore.

Gli esiti principali, aggiungendo `agent/event_rate/timeouts`, dovrebbero
sommare approssimativamente a 1.

### Come usarle per il debug

- Successi fermi e collisioni alte: controllare reward terminali, esplorazione
  e qualita' delle dimostrazioni expert.
- Successi alti ma off-road crescente: probabile scorciatoia appresa o reward
  insufficiente per il rispetto della carreggiata.
- Tutti gli esiti molto variabili: policy instabile, finestre troppo piccole o
  reward model che cambia rapidamente.
- Se la loss del reward model migliora ma gli esiti no, la loss non sta
  producendo un segnale utile per la policy.

## 2. Performance ambientale

Metriche:

- `agent/rewards/ep_fast_return`
- `agent/rewards/ep_comfort_return`

### Cosa rappresentano

`ep_fast_return` e' il return della funzione ambientale orientata alla
performance. `ep_comfort_return` misura invece la qualita' di guida e tende a
essere negativo: valori meno negativi indicano maggiore comfort.

Queste metriche usano il reward ambientale come riferimento diagnostico anche
quando SAC viene allenato con il reward appreso.

### Come interpretarle

- `ep_fast_return` crescente e' positivo solo se collisioni e off-road non
  aumentano.
- `ep_comfort_return` dovrebbe diventare meno negativo senza ridurre troppo
  successi e velocita'.
- Fast return alto e comfort molto negativo indicano guida efficace ma
  aggressiva o oscillatoria.
- I valori assoluti vanno confrontati solo tra run dello stesso ambiente e con
  la stessa funzione di reward.

### Come usarle per il debug

- Fast return crescente con successi fermi: possibile aumento di velocita'
  senza reale miglioramento degli esiti.
- Comfort in peggioramento insieme a collisioni: controllare azioni al limite,
  entropy coefficient e penalita' ambientali.
- Entrambi piatti: controllare che SAC riceva reward non costanti e che il
  replay relabelling sia operativo.

## 3. Ranking dei return

Metriche nello stesso pannello:

- `reward_val/current_rollout/post_update/spearman_returns`
- `reward_val/debug_dataset/post_update/spearman_returns`

### Cosa rappresentano

La correlazione di Spearman misura se il reward model ordina le traiettorie
nello stesso modo del true reward. Considera il ranking e non richiede che le
due scale numeriche coincidano.

- `current_rollout`: traiettorie prodotte dalla policy corrente.
- `debug_dataset`: insieme fisso, bilanciato e confrontabile fra iterazioni.
- `post_update`: valutazione dopo l'update del reward model.

### Come interpretarle

- `+1`: ranking perfettamente concorde.
- `0`: ranking sostanzialmente casuale.
- Valore negativo: il modello tende a preferire le traiettorie peggiori.
- Current e debug entrambi positivi e crescenti: apprendimento generalizzabile.
- Current positivo e debug vicino a zero o negativo: adattamento ai rollout
  recenti senza generalizzazione.

### Come usarle per il debug

- Debug Spearman fermo vicino a zero: verificare qualita' degli expert, loss,
  length bias e capacita' del reward model.
- Forte differenza current-debug: distribution shift o overfitting al rollout
  corrente.
- Oscillazioni di segno: reward non stazionario, dataset troppo piccolo o
  update del reward model troppo aggressivi.
- Confrontare occasionalmente `pre_update` e `post_update`: se migliora solo il
  current rollout, l'update sta memorizzando la distribuzione corrente.

## 4. Separazione tra arrivo e collisione

Metriche nello stesso pannello:

- `reward_val/current_rollout/post_update/gap_arrived_collided`
- `reward_val/debug_dataset/post_update/gap_arrived_collided`

### Cosa rappresentano

Il gap e' calcolato come:

```text
reward medio sulle transizioni di arrivo
- reward medio sulle transizioni di collisione
```

Misura direttamente se il reward model assegna un valore terminale maggiore
agli arrivi rispetto alle collisioni.

### Come interpretarle

- Gap positivo: ordinamento semanticamente corretto.
- Gap vicino a zero: il modello non distingue i due esiti.
- Gap negativo: errore grave; la collisione viene premiata piu' dell'arrivo.
- Current positivo e debug negativo: separazione dipendente dalla distribuzione
  corrente e non generalizzata.

Il gap misura solo la separazione terminale. Va letto insieme a Spearman, che
considera l'intera traiettoria.

### Come usarle per il debug

- Gap debug negativo: controllare subito gli expert con collisione, le feature
  di status e il bilanciamento dei dati.
- Gap che cambia segno frequentemente: update troppo forti o reward model
  sensibile a pochi campioni terminali.
- Gap alto ma Spearman basso: il terminale e' classificato correttamente, ma il
  reward durante gli stati `running` non ordina bene le traiettorie.
- Gap e Spearman entrambi negativi: non usare quel checkpoint per addestrare o
  valutare la policy.

## 5. Stabilita' del reward model

Metriche:

- `reward/loss`
- `reward/weight_norm`
- `reward/grad_norm`

### Cosa rappresentano

- `reward/loss`: obiettivo usato per allenare il reward model. Con le loss
  MaxEnt puo' essere negativo e non deve necessariamente convergere a zero.
- `reward/weight_norm`: norma L2 complessiva dei parametri dell'ensemble.
- `reward/grad_norm`: intensita' media dei gradienti durante gli update.

### Come interpretarle

- Tutti i valori devono rimanere finiti.
- Loss decrescente senza miglioramento di Spearman e gap non rappresenta un
  progresso utile.
- Weight norm in crescita continua suggerisce scale drift, soprattutto con
  `l2_rew=0`.
- Gradient norm con picchi ripetuti puo' indicare update instabili; una
  diminuzione graduale e' generalmente normale.

La normalizzazione applicata ai reward di SAC puo' proteggere l'agente dalla
scala, ma non corregge un ranking semanticamente sbagliato.

### Come usarle per il debug

- Loss o gradient norm non finite: fermare la run e controllare learning rate,
  input e traiettorie anomale.
- Weight norm crescente e return scale crescente: aggiungere regolarizzazione
  L2 o ridurre `lr_rew`/`gradient_steps_rew`.
- Gradient norm alto ma weight norm stabile: cercare batch anomali o outlier.
- Loss stabile ma metriche di validazione peggiori: overfitting o loss non
  allineata all'obiettivo reale.

## 6. Diagnostiche MaxEnt-2

Metriche:

- `reward/maxent2_effective_sample_fraction`
- `reward/maxent2_expert_softmax_mass`

### Cosa rappresentano

`effective_sample_fraction` e' l'Effective Sample Size diviso per il numero di
traiettorie nella partizione. Misura quanto il gradiente sia distribuito sui
campioni.

`expert_softmax_mass` e' la frazione della massa softmax assegnata alle
traiettorie expert dalla partizione storica `maxent_2`.

### Come interpretarle

- ESS fraction vicina a 1: contributi abbastanza uniformi.
- ESS fraction vicina al minimo: pochi campioni dominano l'update.
- Sotto 0.1 e' un segnale di collasso della stima.
- Expert mass molto alta significa che la partizione e il gradiente sono
  dominati dagli expert invece che dalle traiettorie della policy.
- Una expert mass alta non dimostra da sola che il modello abbia imparato un
  buon reward.

### Come usarle per il debug

- ESS bassa: aumentare batch, ridurre la scala dei return o usare una loss piu'
  stabile.
- ESS buona ma Spearman basso: il problema non e' il collasso su un campione,
  ma l'obiettivo o la qualita' dei dati.
- Expert mass persistentemente vicina a 1: verificare scale expert/model e
  considerare `maxent_corrected`.
- Expert mass crescente insieme a gap debug negativo: controllare se il
  dataset expert contiene esiti indesiderati trattati come dimostrazioni
  positive.

Per `maxent` o `maxent_corrected` usare le metriche equivalenti con prefisso
`reward/maxent_*` o `reward/maxent_corrected_*`.

## 7. Non stazionarieta' del replay buffer

Metriche nello stesso pannello:

- `replay_relabel_debug/staleness_ratio`
- `replay_relabel_debug/stored_current_corr`
- `replay_relabel_debug/sign_flip_frac`

### Cosa rappresentano

- `staleness_ratio`: differenza assoluta media fra reward memorizzato e reward
  corrente, normalizzata per la deviazione standard corrente.
- `stored_current_corr`: correlazione tra reward storico e corrente sugli stessi
  campioni del replay buffer.
- `sign_flip_frac`: frazione di campioni per cui il reward cambia segno.

Con relabelling attivo SAC usa il reward corrente durante il sampling del
replay. Le metriche descrivono comunque quanto il target del critic cambi nel
tempo.

### Come interpretarle

- Staleness verso 0 e correlazione verso 1: reward model in stabilizzazione.
- Staleness sopra 1: il cambiamento e' grande rispetto alla variabilita' utile
  del reward corrente.
- Sign flip verso 0: pochi cambiamenti qualitativi di valutazione.
- Correlazione bassa con sign flip basso: possibile trasformazione non lineare
  o forte riordinamento senza cambio di segno.

### Come usarle per il debug

- Staleness alta e critic loss alta: il critic insegue target troppo mobili.
- Correlazione bassa: ridurre aggressivita' degli update del reward model o
  accorciare la storia effettiva del replay buffer.
- Molti sign flip: controllare normalizzazione, scale drift e cambiamenti dei
  reward terminali.
- Metriche pessime con `relabel_rewards=False`: il critic sta realmente usando
  reward obsoleti; confrontare con l'ablation `relabel_rewards=True`.

## 8. Stabilita' di SAC

Metriche:

- `agent/train/critic_loss`
- `agent/train/ent_coef`

### Cosa rappresentano

`critic_loss` misura l'errore dei critic rispetto ai TD target. `ent_coef` e'
il coefficiente di entropia `alpha`, adattato automaticamente da SAC, che
regola il compromesso fra esplorazione e sfruttamento.

### Come interpretarle

- La critic loss deve restare finita; non e' necessario che vada a zero.
- Va letta rispetto alla scala e alla non stazionarieta' del reward.
- `ent_coef` deve restare positivo. Puo' diminuire quando la policy diventa
  piu' deterministica.
- Un collasso rapido verso zero puo' indicare perdita prematura di esplorazione.

### Come usarle per il debug

- Critic loss crescente insieme a staleness e weight norm: reward target non
  stazionari o scale drift.
- Critic loss alta ma performance in crescita: puo' essere soprattutto un
  effetto della scala; non diagnosticare divergenza dalla loss isolata.
- Entropy coefficient quasi zero e policy bloccata: aumentare esplorazione o
  controllare il target entropy.
- Entropy alta e risultati molto variabili: la policy potrebbe non riuscire a
  stabilizzarsi.

## 9. Imitazione state-action

Metrica:

- `imitation/state_action_auc`

### Cosa rappresenta

E' l'AUC su validation set di una logistic regression addestrata da zero a
distinguere transizioni expert e agent usando la concatenazione di
`observation` e `action`. Lo split viene effettuato per traiettoria e le due
classi sono bilanciate, quindi traiettorie piu' lunghe non determinano da sole
il risultato.

E' un classifier two-sample test della occupancy state-action. Il
classificatore e' soltanto diagnostico: non modifica reward model o policy.

### Come interpretarla

- `0.5`: expert e agent non sono distinguibili dal classificatore lineare.
- `0.5-0.6`: differenza debole.
- `0.6-0.75`: differenza moderata da monitorare.
- Sopra `0.75`: occupancy chiaramente differenti.
- Vicina a `1`: expert e agent sono quasi perfettamente separabili.

Se l'obiettivo e' imitare il comportamento expert, la curva dovrebbe scendere
verso `0.5`. Un success rate alto insieme ad AUC alta significa che l'agente
risolve il task, ma con stati visitati o azioni differenti dall'expert.

Un valore vicino a `0.5` non prova che le distribuzioni siano identiche: indica
solo che questa logistic regression non riesce a separarle. Un valore alto e'
invece una forte evidenza di differenza. La metrica viene normalmente calcolata
ogni `imitation_diagnostics_interval` iterazioni; una linea fra due punti W&B
non dimostra che il trend intermedio sia lineare.

### Come usarla per il debug

- AUC alta e successi bassi: l'agente non ha ancora imparato il task.
- AUC alta e successi alti: comportamento efficace ma non expert-like.
- AUC piatta: controllare preprocessing, scala delle azioni, condizioni
  iniziali e copertura del dataset expert.
- Prima di cambiare algoritmo, verificare come sanity check che expert-vs-expert,
  agent-vs-agent e label casuali producano AUC vicine a `0.5`.
- Per localizzare la differenza, confrontare separatamente distribuzioni degli
  stati, delle azioni, velocita' e comfort.

## Ordine consigliato per il debug

1. Controllare esiti e fast/comfort return per stabilire se la policy migliora
   nel mondo reale.
2. Controllare `imitation/state_action_auc` per distinguere successo nel task
   e reale imitazione dell'occupancy expert.
3. Controllare Spearman e gap sul debug dataset per verificare che il reward
   appreso sia semanticamente corretto.
4. Confrontare current rollout e debug dataset per identificare overfitting o
   distribution shift.
5. Controllare staleness, correlazione e sign flip per misurare quanto il
   reward cambi nel replay buffer.
6. Usare loss, norme, ESS e massa expert per trovare la causa nel reward model.
7. Controllare critic loss ed entropy coefficient per verificare l'effetto
   finale su SAC.

Non scegliere un checkpoint usando soltanto `reward/loss` o
`agent/rollout/ep_rew_mean`: il primo e' un obiettivo di ottimizzazione, mentre
il secondo usa il reward appreso. Le metriche principali per la selezione sono
gli esiti ambientali, l'AUC state-action, Spearman e
`gap_arrived_collided` sul debug dataset.
