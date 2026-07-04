Task: Integrazione di dimostrazioni esperte e preferenze in human_feedback_rl

Obiettivo

Implementare un framework che integri due sorgenti di feedback:

1. Dimostrazioni esperte, tramite maxent_2 loss.
2. Preferenze umane, tramite l’algoritmo già presente nella repository.

L’obiettivo è addestrare un reward/cost model che sfrutti entrambe le sorgenti di informazione e verificare sperimentalmente che la combinazione di dimostrazioni + preferenze produca prestazioni migliori rispetto all’uso delle singole sorgenti separatamente.

Contesto

Nella repository human_feedback_rl è già presente un algoritmo che utilizza feedback di preferenze per addestrare un reward model.

È stata richiesta anche l’implementazione di maximum entropy per usare traiettorie esperte nell’ambiente SUMO.

Ora bisogna implementare l’integrazione tra questi due approcci, in modo da creare un framework unificato per l’apprendimento del reward/cost model da feedback eterogeneo.

Analizzare l’implementazione già presente nella repository relativa al training da preferenze, per capire:

* come vengono rappresentate le preferenze;
* come viene addestrato il reward model;
* come viene aggiornata la policy;
* come vengono gestiti logging, configurazioni, launcher e checkpoint.

Dataset e modello esperto

Le traiettorie esperte sono disponibili in:

datasets/expert_trajectories.pkl

Versione senza collisioni:

datasets/expert_trajectories_no_collision.pkl

Il modello esperto PPO si trova in:

sumo-rl-ego/sumo_rl_ego/policies/models/ppo-fast

Obiettivo tecnico

Implementare un algoritmo combinato che permetta di addestrare un reward/cost model usando contemporaneamente:

* una loss da dimostrazioni esperte, basata su Guided Cost Learning;
* una loss da preferenze, basata sull’algoritmo già esistente nella repository.

Il training deve consentire tre modalità sperimentali:

1. Solo preferenze
2. Solo dimostrazioni
3. Dimostrazioni + preferenze

Queste modalità devono essere selezionabili da configurazione.

Cosa implementare

Implementare almeno:

* caricamento delle traiettorie esperte;
* caricamento o generazione dei dati di preferenza secondo il codice esistente;
* reward/cost model condiviso tra le due sorgenti di feedback;
* loss per Guided Cost Learning;
* loss per preferenze;
* combinazione pesata delle due loss;
* training della policy sul reward/cost model appreso;
* ciclo completo di training;
* evaluation;
* logging;
* checkpoint.

La loss complessiva deve avere una forma configurabile simile a:

total_loss = lambda_demo * gcl_loss + lambda_pref * preference_loss + regularization_terms

dove lambda_demo e lambda_pref devono essere iperparametri configurabili.

Modalità sperimentali richieste

La configurazione deve permettere di eseguire almeno:

1. Preference-only

Usa solo il feedback di preferenze.

lambda_demo = 0
lambda_pref > 0

2. Demonstration-only

Usa solo le dimostrazioni esperte.

lambda_demo > 0
lambda_pref = 0

3. Hybrid feedback

Usa sia dimostrazioni esperte sia preferenze.

lambda_demo > 0
lambda_pref > 0

Policy optimization

La policy deve essere addestrata usando il reward/cost model appreso.

Se nella repository esiste già una procedura di policy optimization riutilizzabile, può essere usata. In caso contrario, implementare una soluzione semplice, modulare e documentata.

Documentare chiaramente eventuali differenze rispetto all’algoritmo originale Guided Cost Learning o rispetto all’algoritmo preference-based già esistente.

Logging su Weights & Biases

Integrare logging completo su Weights & Biases.

Registrare almeno:

* fast_return: true return dell’ambiente;
* success_rate;
* expert_action_rmse: RMSE tra le azioni della policy e quelle dell’esperto;
* gcl_loss;
* preference_loss;
* total_reward_model_loss;
* policy_loss;
* lambda_demo;
* lambda_pref;
* metriche separate per le tre modalità sperimentali;
* eventuali metriche aggiuntive utili.

File da realizzare

Creare almeno:

* uno script principale per il training ibrido;
* un launcher per eseguire l’esperimento;
* un file di configurazione per impostare modalità sperimentale, pesi delle loss e iperparametri;
* eventuali moduli separati per dataset, reward model, loss GCL, loss preferenze, policy training ed evaluation.

La struttura dei file è libera, ma deve essere chiara, modulare ed estendibile.

Vincoli

* Non modificare file non necessari.
* Non rompere il codice esistente basato su preferenze.
* Non eliminare o sovrascrivere gli algoritmi già presenti.
* È possibile riutilizzare codice esistente quando utile, ma non è obbligatorio seguire rigidamente la struttura attuale.

Verifiche finali

Prima di considerare conclusa l’implementazione:

* verificare che la modalità solo preferenze funzioni;
* verificare che la modalità solo dimostrazioni funzioni;
* verificare che la modalità ibrida funzioni;
* verificare che il reward/cost model riceva correttamente entrambe le loss;
* verificare che lambda_demo e lambda_pref controllino effettivamente il contributo delle due sorgenti;
* verificare che il logging su Weights & Biases distingua chiaramente le tre modalità;
* verificare che i checkpoint vengano salvati correttamente;
* verificare che il training sia eseguibile da riga di comando.

Documentazione

Realizzare una documentazione in formato HTML.

La documentazione deve includere:

* descrizione del framework ibrido;
* spiegazione di come vengono integrate dimostrazioni e preferenze;
* formula della loss complessiva;
* descrizione delle tre modalità sperimentali;
* istruzioni per eseguire training ed evaluation;
* descrizione dei file creati;
* descrizione degli iperparametri principali;
* eventuali adattamenti rispetto ai paper o al codice esistente;
* limitazioni note e possibili sviluppi futuri.

Obiettivo finale

L’implementazione deve permettere di confrontare sperimentalmente:

* feedback da preferenze;
* feedback da dimostrazioni;
* feedback ibrido preferenze + dimostrazioni.

Il risultato atteso è un framework modulare che consenta di mostrare che l’integrazione di più forme di feedback produce risultati migliori rispetto all’utilizzo di una singola sorgente.