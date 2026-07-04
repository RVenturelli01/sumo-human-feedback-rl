# Guida ai grafici W&B di Demo SAC

Questa guida descrive le metriche presenti nella run
[`andrea02polimi-politecnico-di-milano/demo-sac/4c11yj2o`](https://wandb.ai/andrea02polimi-politecnico-di-milano/demo-sac/runs/4c11yj2o),
cosa misurano, a cosa servono e quale andamento e' desiderabile.

La run usa SAC, loss `maxent_corrected`, reward relabelling attivo, un ensemble
di 3 reward model e batch di 64 traiettorie expert e model. La run era ancora
in esecuzione durante l'analisi; la sezione finale e' quindi una fotografia
intermedia, non una valutazione conclusiva.

## Come leggere gli assi

| Famiglia | Asse X corretto | Significato |
|---|---|---|
| `agent/*` | `agent/time/total_timesteps` | Transizioni ambientali viste da SAC. |
| `reward/*` | `iterations` | Alternanze complete: rollout, update reward model, update SAC. |
| `reward_val/*` | `iterations` | Snapshot del reward model prima e dopo ogni update. |
| `replay_relabel_debug/*` | `iterations` | Stato del replay buffer a ogni iterazione. |
| `rollout/*`, `time/*` | `iterations` | Raccolta dati e tempi dell'iterazione. |

Non usare `_step` come asse: e' un contatore interno W&B che mescola i dump di
SAC e quelli del ciclo esterno.

Le metriche marcate come nascoste non sono cancellate. W&B le salva nella
history, ma non crea automaticamente un pannello per ciascuna di esse.

## Concetti essenziali

### `pre_update` e `post_update`

- `pre_update`: reward model valutato sui dati appena raccolti, prima di essere
  allenato in quella iterazione.
- `post_update`: stesso insieme di dati valutato dopo l'update del reward model.
- Il confronto pre/post misura l'effetto immediato dei gradient step.
- Un miglioramento solo sul current rollout e non sul debug dataset suggerisce
  overfitting ai dati recenti.

### `current_rollout` e `debug_dataset`

- `current_rollout`: distribuzione generata dalla policy corrente; cambia a ogni
  iterazione ed e' utile per rilevare problemi on-policy e nuovi stati OOD.
- `debug_dataset`: insieme fisso; rende confrontabili le iterazioni e misura la
  generalizzazione del reward model.

### Reward ambientale e reward appreso

Il reward appreso puo' cambiare scala e offset senza cambiare necessariamente
il comportamento preferito. Per questo i suoi valori assoluti non vanno letti
come una normale curva di performance.

In questa pipeline:

- `agent/rollout/ep_rew_mean` e `agent/rewards/ep_env_return` seguono il reward
  consegnato a SAC, cioe' il reward appreso;
- `agent/rewards/ep_fast_return`, gli event rate e
  `rollout/mean_true_reward` sono riferimenti migliori per la performance reale.

## 1. Performance della policy

Questi sono i primi grafici da controllare per capire se SAC sta imparando un
comportamento utile, indipendentemente dalla loss del reward model.

| Metrica | Cosa mostra e a cosa serve | Comportamento atteso |
|---|---|---|
| `agent/event_rate/successes` | Frazione di episodi terminati con arrivo. E' la metrica di successo principale. | Crescita e successiva stabilizzazione verso 1. |
| `agent/event_rate/collisions` | Frazione di episodi terminati con collisione. | Discesa verso 0. Va letta insieme al successo. |
| `agent/event_rate/off_road` | Frazione di episodi terminati fuori strada. | Vicina a 0. Una crescita segnala una policy aggressiva o instabile. |
| `agent/event_rate/timeouts` | Frazione di episodi terminati per timeout. | Vicina a 0. Un valore alto indica policy troppo lenta, bloccata o indecisa. |
| `agent/rewards/ep_fast_return` | Return della funzione ambientale `fast`, calcolato come metrica anche quando SAC usa il reward appreso. | Crescita senza aumento di collisioni o perdita di comfort. |
| `agent/rewards/ep_comfort_return` | Return della funzione di comfort. | Dovrebbe aumentare, cioe' diventare meno negativo, senza sacrificare successo e velocita'. |
| `agent/rewards/ep_env_return` | Return episodico visto dal wrapper SAC; con `EnvRewardWrapper` coincide sostanzialmente con il reward appreso. | Non ha un target assoluto. Deve restare finito e va interpretato con le metriche ambientali. |
| `agent/performance/ep_avg_speed` | Velocita' media per episodio. | Avvicinamento alla velocita' utile del task, non crescita illimitata. |
| `agent/performance/ep_length` | Numero medio di step per episodio. | Puo' diminuire con arrivi piu' rapidi; va sempre letto con esito e velocita'. |
| `agent/performance/ep_duration` | Durata simulata media degli episodi. | Simile a `ep_length`; nessuna direzione e' buona da sola. |
| `agent/rollout/ep_len_mean` | Media SB3 della lunghezza degli ultimi episodi. | Coerente con `ep_length`; cambi bruschi richiedono controllo degli esiti. |
| `agent/rollout/ep_rew_mean` | Media SB3 del return usato per allenare SAC, quindi reward appreso. | Finita e non esplosiva. Non usarla da sola per scegliere il modello migliore. |

Le quattro metriche di evento dovrebbero essere quasi esaustive: in condizioni
normali `successes + collisions + off_road + timeouts` e' circa 1.

## 2. Distribuzione delle azioni

Queste metriche sono nascoste dai pannelli automatici, ma rimangono utili quando
la policy sembra bloccata o sfrutta scorciatoie.

| Metrica | Cosa mostra | Comportamento atteso |
|---|---|---|
| `agent/action_rate/acc` | Frazione di step con accelerazione longitudinale sopra soglia. | Non deve saturare stabilmente a 0 o 1; dipende dal traffico. |
| `agent/action_rate/dec` | Frazione di step con decelerazione sotto soglia. | Presente quando serve, ma non dominante senza motivo. |
| `agent/action_rate/ss` | Frazione di step con comando longitudinale circa costante. | Valore intermedio e stabile e' plausibile in cruising. |
| `agent/action_rate/lcl` | Frequenza di comandi di cambio corsia a sinistra. | Bassa ma non necessariamente zero; picchi possono indicare oscillazione. |
| `agent/action_rate/lcr` | Frequenza di comandi di cambio corsia a destra. | Come `lcl`; controllare forte asimmetria non spiegata dallo scenario. |

Per l'azione continua, `ss + acc + dec` dovrebbe essere circa 1. `lcl` e `lcr`
sono calcolate separatamente sul comando laterale.

## 3. Ottimizzazione SAC

| Metrica | Cosa mostra e a cosa serve | Comportamento atteso |
|---|---|---|
| `agent/train/actor_loss` | Obiettivo dell'attore SAC. Include Q-value ed entropia. | Finito; non e' richiesto che vada a zero. Deriva monotona estrema puo' riflettere reward/Q-value in crescita. |
| `agent/train/critic_loss` | Errore dei critic sui TD target. | Finito e possibilmente stabile o decrescente. Va rapportato alla scala del reward. |
| `agent/train/ent_coef` | Coefficiente di entropia `alpha`, appreso con `ent_coef=auto`. | Positivo e non collassato immediatamente a zero; puo' scendere quando la policy diventa piu' deterministica. |
| `agent/train/ent_coef_loss` | Loss usata per adattare `alpha`. | Finita e oscillante attorno a valori piccoli quando l'entropia e' vicina al target. |
| `agent/train/learning_rate` | Learning rate corrente di SAC. | Costante in questa configurazione; varia solo con uno schedule. |
| `agent/train/n_updates` | Numero cumulativo di update SAC. | Crescita monotona. Serve a verificare il rapporto dati/update. |

Una critic loss crescente insieme a reward scale, actor loss e Q-value sarebbe
un segnale di scale drift. La sola critic loss non dimostra divergenza.

## 4. Contatori e throughput SAC

| Metrica | Cosa mostra | Comportamento atteso |
|---|---|---|
| `agent/time/total_timesteps` | Numero cumulativo di transizioni. E' l'asse X delle metriche agent. | Crescita monotona. |
| `agent/time/episodes` | Episodi completati. | Crescita monotona; la pendenza dipende dalla durata degli episodi. |
| `agent/time/fps` | Transizioni elaborate al secondo secondo SB3. | Abbastanza stabile; cali persistenti indicano rallentamenti di SUMO o logging. |
| `agent/time/time_elapsed` | Tempo trascorso riportato nella singola chiamata `learn`. | Crescita durante ogni chiamata; puo' ripartire tra iterazioni. |

Queste metriche sono nascoste automaticamente per non occupare pannelli; il
totale timestep rimane comunque l'asse delle curve SAC.

## 5. Rollout usato dal reward model

| Metrica | Cosa mostra e a cosa serve | Comportamento atteso |
|---|---|---|
| `rollout/mean_true_reward` | Return ambientale medio delle traiettorie raccolte. | Crescita o stabilizzazione su valori buoni; e' una metrica di policy importante. |
| `rollout/mean_model_reward` | Somma media dei reward predetti sulle stesse traiettorie. | Finita; la scala puo' cambiare. Deve essere coerente con ranking e performance, non necessariamente con il true return in valore assoluto. |
| `rollout/mean_length` | Lunghezza media delle traiettorie raccolte. | Stabile o coerente con un aumento degli arrivi; non ha direzione autonoma. |
| `rollout/action_at_bound_fraction` | Frazione di transizioni in cui almeno una componente dell'azione e' esattamente al limite. | Generalmente bassa. Valori alti indicano saturazione della policy. |
| `rollout/action_component_at_bound_fraction` | Frazione di singole componenti d'azione al limite. | Bassa; aiuta a distinguere saturazione parziale da saturazione completa. |

## 6. Loss e scala del reward model

| Metrica | Cosa mostra e a cosa serve | Comportamento atteso |
|---|---|---|
| `reward/loss` | Loss MaxEnt diagnostica: termine expert piu' stima della partizione. Puo' essere negativa. | Finita e senza oscillazioni esplosive. Non deve necessariamente tendere a zero. |
| `reward/expert_return_mean` | Return predetto medio sulle traiettorie expert. | Idealmente maggiore del model return, ma senza crescita incontrollata della scala. |
| `reward/model_return_mean` | Return predetto medio sulle traiettorie della policy corrente. | Puo' avvicinarsi all'expert quando la policy migliora. |
| `reward/expert_model_margin` | `expert_return_mean - model_return_mean`. | Positivo durante l'apprendimento; puo' ridursi se la policy raggiunge l'expert. Un aumento dovuto solo alla scala non e' progresso. |
| `reward/return_std` | Deviazione standard dei return expert e model usati nella diagnostica. | Non deve esplodere. Crescita continua suggerisce scale drift. |
| `reward/return_abs_mean` | Ampiezza media assoluta dei return. Nascosta automaticamente. | Stabile; crescita monotona senza miglioramento di ranking e' sospetta. |
| `reward/return_min` | Return minimo nel batch diagnostico. Nascosta automaticamente. | Finito e senza outlier sempre piu' estremi. |
| `reward/return_max` | Return massimo nel batch diagnostico. Nascosta automaticamente. | Finito e senza outlier sempre piu' estremi. |
| `reward/grad_norm` | Norma L2 media dei gradienti nei gradient step e nei membri dell'ensemble. | Tendenza a stabilizzarsi o diminuire; picchi isolati sono possibili. |
| `reward/grad_norm_max` | Massimo gradient norm osservato nell'update. Nascosta automaticamente. | Non deve crescere ripetutamente o diventare non finito. Evidenzia spike persi dalla media. |
| `reward/weight_norm` | Norma L2 complessiva dei parametri dell'ensemble. | Plateau ragionevole. Crescita continua indica che il modello sta aumentando la scala dei reward. |

La run usa `l2_rew=0`, quindi `weight_norm` e le statistiche di scala meritano
particolare attenzione.

## 7. Diagnostiche `maxent_corrected`

La partizione usa i logit corretti `R(tau) / temperature - log q(tau)`. I pesi
softmax risultanti determinano quali traiettorie model contribuiscono al
gradiente.

| Metrica | Cosa mostra e a cosa serve | Comportamento atteso |
|---|---|---|
| `reward/maxent_corrected_partition_model` | Stima log-sum-exp della partizione sulle traiettorie model. | Finita e su scala compatibile con gli expert return. Il valore assoluto non e' una performance. |
| `reward/maxent_corrected_log_q_mean` | Log-probabilita' media delle traiettorie sotto la policy proposta. | Finita e relativamente stabile. Va interpretata insieme a return e lunghezza delle traiettorie. |
| `reward/maxent_corrected_top1_softmax_weight` | Peso della singola traiettoria piu' influente. | Lontano da 1. Valori prossimi a 1 indicano collasso della stima su un campione. |
| `reward/maxent_corrected_top5_softmax_mass` | Massa totale assegnata alle cinque traiettorie piu' influenti. | Non dovrebbe essere stabilmente vicina a 1 con batch 64. |
| `reward/maxent_corrected_effective_sample_size` | Numero effettivo di traiettorie che contribuiscono: `1 / sum(w^2)`. | Ben maggiore di 1; massimo pari al numero di traiettorie model disponibili. |
| `reward/maxent_corrected_effective_sample_fraction` | ESS diviso per il numero di traiettorie. Rende confrontabili batch diversi. | Preferibilmente sopra 0.1. Il minimo con batch 64 e' `1/64 = 0.015625`. |

La coppia ESS fraction/top-1 weight e' piu' informativa della sola loss. Una
loss finita non impedisce che quasi tutto il gradiente provenga da una sola
traiettoria.

## 8. Replay buffer e reward relabelling

Il buffer conserva il reward calcolato al momento dell'inserimento. La
diagnostica lo confronta con la predizione del reward model corrente sullo
stesso campione.

Con relabelling attivo il buffer non riscrive fisicamente i valori storici:
ricalcola il reward quando SAC estrae il batch. Quindi una forte staleness puo'
comparire nei grafici anche se il critic usa correttamente reward aggiornati.

| Metrica | Cosa mostra e a cosa serve | Comportamento atteso |
|---|---|---|
| `replay_relabel_debug/relabel_enabled` | Flag 1/0 della configurazione. Nascosto automaticamente. | Costante a 1 in questa run. |
| `replay_relabel_debug/critic_uses_current_reward` | Conferma che il batch SAC riceve reward correnti. Nascosto automaticamente. | Costante a 1 con relabelling attivo. |
| `replay_relabel_debug/sample_size` | Numero di transizioni usate nella diagnostica. Nascosto automaticamente. | Generalmente 2048 dopo il riempimento iniziale. |
| `replay_relabel_debug/stored_reward_mean` | Media dei reward storici memorizzati. Nascosta automaticamente. | Dovrebbe avvicinarsi alla current mean quando il reward model si stabilizza. |
| `replay_relabel_debug/current_reward_mean` | Media delle predizioni correnti sugli stessi campioni. Nascosta automaticamente. | Stabile e progressivamente vicina alla stored mean. |
| `replay_relabel_debug/stored_reward_std` | Dispersione dei reward memorizzati. Nascosta automaticamente. | Compatibile con la current std quando il modello smette di cambiare. |
| `replay_relabel_debug/current_reward_std` | Dispersione delle predizioni correnti. Nascosta automaticamente. | Finita e senza crescita incontrollata. |
| `replay_relabel_debug/delta_mean` | Media di `current - stored`. Nascosta automaticamente. | Verso 0; il segno indica la direzione media del drift. |
| `replay_relabel_debug/delta_std` | Dispersione del delta. Nascosta automaticamente. | Diminuzione quando il reward model si stabilizza. |
| `replay_relabel_debug/delta_abs_mean` | Differenza assoluta media tra reward corrente e storico. | Tendenza a diminuire. |
| `replay_relabel_debug/delta_abs_p95` | 95-esimo percentile della differenza assoluta. | Tendenza a diminuire; rileva una coda di campioni molto obsoleti. |
| `replay_relabel_debug/staleness_ratio` | `mean(abs(delta)) / std(current_reward)`. Misura il drift rispetto alla scala corrente. | Idealmente sotto 1 e in diminuzione; sopra 1 significa drift grande quanto o piu' della variabilita' utile. |
| `replay_relabel_debug/sign_flip_frac` | Frazione di transizioni in cui reward storico e corrente hanno segno opposto. | Verso 0. Valori alti indicano cambiamenti qualitativi, non solo di scala. |
| `replay_relabel_debug/stored_current_corr` | Correlazione tra reward storico e corrente. | Verso 1. Una correlazione bassa indica cambiamento del ranking locale. |

Per l'ablation, il confronto piu' importante e' tra run con `relabel_enabled=1`
e `0` a parita' di seed. Le metriche di staleness descrivono il mismatch; le
metriche ambientali mostrano se quel mismatch danneggia davvero SAC.

## 9. Validazione del reward model

Tutte le metriche seguenti esistono per le quattro combinazioni:

```text
reward_val/current_rollout/pre_update/<metrica>
reward_val/current_rollout/post_update/<metrica>
reward_val/debug_dataset/pre_update/<metrica>
reward_val/debug_dataset/post_update/<metrica>
```

La tabella descrive il suffisso `<metrica>` una volta sola.

| Suffisso | Cosa mostra e a cosa serve | Comportamento atteso |
|---|---|---|
| `reward_mean` | Reward per transizione medio predetto dall'ensemble. | Nessun target assoluto; deve restare finito e relativamente stabile. |
| `reward_std` | Variabilita' delle predizioni fra transizioni. | Non deve collassare senza motivo ne' esplodere. |
| `reward_min` | Minimo reward predetto. Nascosto automaticamente. | Finito; utile per outlier negativi. |
| `reward_max` | Massimo reward predetto. Nascosto automaticamente. | Finito; utile per outlier positivi. |
| `reward_running` | Reward medio quando l'episodio e' ancora in corso. Nascosto automaticamente. | Riferimento per confrontare i terminali. |
| `reward_arrived` | Reward medio sulle transizioni terminali di successo. Nascosto automaticamente. | Maggiore di collisione e, se e' previsto un bonus terminale, maggiore di running. |
| `reward_collided` | Reward medio sulle collisioni. Nascosto automaticamente. | Inferiore ad arrived e running. |
| `reward_offroad` | Reward medio sulle uscite di strada. Nascosto automaticamente. | Basso/negativo rispetto al running. |
| `reward_timeout` | Reward medio sui timeout. Nascosto automaticamente. | Inferiore a un arrivo; la severita' dipende dal task. |
| `gap_arrived_collided` | `reward_arrived - reward_collided`. Riassume la separazione successo/fallimento. | Positivo e robusto sia sul current rollout sia sul debug dataset. |
| `gap_arrived_running` | `reward_arrived - reward_running`. Verifica il bonus relativo dell'arrivo. | Generalmente positivo. Un valore negativo indica che arrivare e' valutato meno di continuare. |
| `ensemble_std` | Disagreement medio fra i membri dell'ensemble. Proxy di incertezza/OOD. | Basso e stabile sul debug set; sul current rollout puo' salire quando la policy visita stati nuovi. |
| `ensemble_std_running` | Disagreement limitato agli stati running. Nascosto automaticamente. | Simile o inferiore all'ensemble std totale; crescita persistente indica OOD negli stati ordinari. |
| `spearman_returns` | Correlazione di rango fra true return e model return a livello di traiettoria. | Positiva e crescente; circa 0 significa ranking casuale, 1 ranking perfetto. Sul debug set e' la misura piu' comparabile. |
| `spearman_returns_defined` | Flag che vale 1 se Spearman e' calcolabile. Nascosto automaticamente. | Sempre 1. Zero indica troppo poche traiettorie o return costanti. |

### Lettura congiunta pre/post

- `post Spearman > pre Spearman`: l'update migliora il ranking sul batch.
- `post current migliora`, ma `post debug peggiora`: probabile overfitting.
- `post ensemble_std` molto maggiore del pre: l'update aumenta il disaccordo.
- Gap positivo sul current e circa zero sul debug: separazione non generalizzata.

## 10. Tempi di esecuzione

| Metrica | Cosa mostra | Comportamento atteso |
|---|---|---|
| `time/sample_rollout` | Secondi impiegati per raccogliere le traiettorie. | Relativamente stabile; cresce con step, durata degli episodi e carico SUMO. |
| `time/train_reward_model` | Secondi per allenare l'ensemble reward. | Stabile a configurazione invariata. |
| `time/train_agent` | Secondi per gli update e la raccolta interna di SAC. | Stabile; spesso e' la componente dominante. |
| `time/loggings` | Overhead delle diagnostiche e del logging. Nascosto automaticamente. | Molto minore delle altre componenti. |
| `time/total` | Durata complessiva dell'iterazione. | Circa somma delle componenti, senza crescita progressiva inattesa. |

## Dashboard consigliato

Per una lettura rapida conviene mantenere cinque sezioni.

### A. Policy reale

- `agent/event_rate/successes`
- `agent/event_rate/collisions`
- `agent/rewards/ep_fast_return`
- `agent/rewards/ep_comfort_return`
- `rollout/mean_true_reward`

### B. Qualita' del reward

- Spearman pre/post su current rollout e debug dataset
- `gap_arrived_collided` pre/post sui due dataset
- `gap_arrived_running` sul debug dataset
- `ensemble_std` pre/post sui due dataset

### C. Stabilita' MaxEnt

- `effective_sample_fraction`
- `top1_softmax_weight`
- `top5_softmax_mass`
- `reward/grad_norm`
- `reward/weight_norm`

### D. Replay relabelling

- `stored_current_corr`
- `sign_flip_frac`
- `staleness_ratio`
- `delta_abs_mean` e `delta_abs_p95`

### E. SAC e runtime

- `agent/train/critic_loss`
- `agent/train/actor_loss`
- `agent/train/ent_coef`
- `time/train_agent`, `time/train_reward_model`, `time/sample_rollout`

## Fotografia intermedia della run `4c11yj2o`

La lettura API ha osservato 8 iterazioni complete, fino a circa 180k timestep.
La run era ancora attiva.

### Segnali positivi

- Il true rollout return e' passato dal valore iniziale circa `-96.7` a valori
  generalmente positivi nelle iterazioni successive.
- La correlazione stored/current del replay e' salita da circa `0.20` a `0.58`.
- La sign-flip fraction e' scesa da circa `0.30` a `0.16` dopo un picco iniziale.
- Il disagreement dell'ensemble e' rimasto sotto 1 negli ultimi snapshot.
- Il relabelling e' attivo, quindi il critic usa reward ricalcolati al sampling.

### Segnali critici da seguire

- L'ESS fraction finale osservata e' circa `0.0186`; con batch vicino a 64 e'
  quasi il minimo teorico. Il top-1 weight e' circa `0.91` ed e' stato spesso 1.
  La partizione MaxEnt e' quindi dominata da una singola traiettoria.
- Lo Spearman sul debug dataset e' ancora basso, circa `0.26`, mentre quello sul
  current rollout e' piu' alto ma molto variabile. Il ranking non generalizza
  ancora in modo convincente.
- Il gap arrived/collided sul debug dataset e' praticamente zero nelle ultime
  iterazioni: il reward model non distingue robustamente successo e collisione
  sul dataset fisso.
- Lo staleness ratio e' circa `1.40`: il reward model continua a cambiare molto
  rispetto ai reward storici. Il relabelling protegge il critic, ma il segnale
  indica che il reward non si e' ancora stabilizzato.
- Gradient norm e loss sono stati molto variabili. Devono essere monitorati
  insieme a ESS, weight norm e ranking, non isolatamente.

La priorita' diagnostica per questa run e' quindi: prima il collasso dei pesi
`maxent_corrected`, poi la generalizzazione sul debug dataset, infine la
stabilizzazione del reward rispetto al replay.
