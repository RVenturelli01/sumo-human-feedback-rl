Task: Implementazione di Guided Cost Learning per SUMO

Obiettivo

Implementare il paper Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization all’interno della repository human_feedback_rl, utilizzando l’ambiente SUMO.

L’obiettivo è aggiungere un algoritmo basato su dimostrazioni esperte che possa successivamente essere integrato con gli altri tipi di feedback già presenti nella repository (ad esempio feedback di preferenze).

⸻

Contesto

Attualmente la repository contiene già un’implementazione di un algoritmo che utilizza feedback di preferenze per apprendere un reward model.

Il nuovo algoritmo deve invece utilizzare dimostrazioni esperte seguendo l’approccio descritto nel paper Guided Cost Learning.

L’obiettivo finale del progetto di ricerca è costruire un framework che integri differenti forme di feedback (preferenze, dimostrazioni esperte, ed eventuali future sorgenti di feedback) e dimostrare sperimentalmente che la loro combinazione produce prestazioni migliori rispetto all’utilizzo di ciascuna sorgente singolarmente.

⸻

Riferimento principale

Prima di iniziare l’implementazione, leggi attentamente il seguente paper:

papers/1603.00448v3.pdf

L’implementazione deve essere il più possibile aderente all’algoritmo descritto nel paper.

Prima di scrivere codice:

1. Leggi completamente il paper.
2. Comprendi l’intero algoritmo Guided Cost Learning.
3. Identifica come adattarlo all’ambiente SUMO.
4. Definisci un breve piano di implementazione.
5. Solo successivamente procedi con l’implementazione.

Se alcuni dettagli implementativi non sono completamente specificati nel paper, scegli la soluzione più coerente con la letteratura e documenta chiaramente ogni assunzione.

Qualsiasi modifica o adattamento necessario per l’ambiente SUMO deve essere esplicitamente documentato.

⸻

Dataset e modello esperto

Le traiettorie esperte sono disponibili in:

datasets/expert_trajectories.pkl

Versione senza collisioni:

datasets/expert_trajectories_no_collision.pkl

Il modello esperto PPO utilizzato per generare le dimostrazioni si trova in:

sumo-rl-ego/sumo_rl_ego/policies/models/ppo-fast

⸻

Vincoli

* Non leggere né utilizzare la cartella reports.
* Non modificare file non necessari all’implementazione.
* Mantieni il codice modulare, leggibile ed estendibile.
* È possibile riutilizzare componenti già presenti nella repository quando risultano utili, ma non è necessario seguire la stessa architettura degli algoritmi esistenti.

⸻

Cosa implementare

Implementare l’intero algoritmo Guided Cost Learning, includendo almeno:

* caricamento delle dimostrazioni esperte;
* implementazione del cost/reward model;
* procedura di ottimizzazione descritta nel paper;
* aggiornamento della policy;
* ciclo completo di training;
* valutazione della policy;
* salvataggio dei checkpoint.

⸻

Logging

Integrare completamente Weights & Biases.

Registrare almeno:

* fast_return (true return dell’ambiente);
* success_rate;
* expert_action_rmse (RMSE tra azioni della policy e azioni dell’esperto);
* loss del cost model;
* loss della policy;
* eventuali metriche aggiuntive utili alla comprensione del training.

⸻

File da realizzare

Creare almeno:

* uno script principale per il training;
* un launcher che richiami lo script principale;
* un file di configurazione contenente gli iperparametri dell’esperimento.

La struttura dei file è libera purché sia chiara e facilmente estendibile.

⸻

Verifiche finali

Prima di considerare conclusa l’implementazione:

* verifica che l’algoritmo implementato corrisponda a quello descritto nel paper;
* controlla che tutte le componenti principali del paper siano state implementate;
* verifica che il training sia eseguibile senza modifiche manuali;
* controlla che il logging su Weights & Biases sia completo;
* verifica che il caricamento delle dimostrazioni esperte funzioni correttamente.

⸻

Documentazione

Realizzare una documentazione in formato HTML.

La documentazione deve includere almeno:

* panoramica dell’algoritmo implementato;
* descrizione dell’architettura del codice;
* corrispondenza tra le sezioni del paper e l’implementazione;
* eventuali adattamenti effettuati per SUMO;
* eventuali differenze rispetto al paper originale e relativa motivazione;
* istruzioni per eseguire training, valutazione e logging;
* descrizione dei principali file creati.

⸻

Obiettivo finale

L’implementazione deve essere sufficientemente modulare da poter essere utilizzata in futuro insieme agli algoritmi basati su feedback di preferenze, così da costruire un framework unificato per l’apprendimento da diverse tipologie di feedback umano.