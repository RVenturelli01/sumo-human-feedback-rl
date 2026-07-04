# Metriche di valutazione per behavioral cloning: Cross-Entropy (proxy KL) e RMSE

## Obiettivo

Due metriche calcolate **solo in fase di eval** (mai come loss di training, nessun gradiente) per monitorare quanto $\pi_a$ sta convergendo verso $\pi_e$ su un validation set held-out.

## 1. Cross-entropy come proxy di KL($\pi_e \| \pi_a$)

### Derivazione

La quantità che vogliamo monitorare è la KL attesa, condizionata sullo stato:

$$
\mathcal{L}_{KL} = \mathbb{E}_{s \sim d_e(s)} \left[ D_{KL}\big(\pi_e(\cdot|s) \,\|\, \pi_a(\cdot|s)\big) \right]
$$

Espandendo la KL:

$$
D_{KL}\big(\pi_e(\cdot|s) \,\|\, \pi_a(\cdot|s)\big) = \underbrace{\sum_a \pi_e(a|s) \log \pi_e(a|s)}_{-H(\pi_e(\cdot|s))} \;-\; \underbrace{\sum_a \pi_e(a|s) \log \pi_a(a|s)}_{\text{cross-entropy}}
$$

Il termine $H(\pi_e)$ è l'entropia dell'esperto: costante, non dipende dai parametri di $\pi_a$, e **non è calcolabile** perché non abbiamo la densità esplicita di $\pi_e$ — solo campioni $(s,a)$.

Quindi:

$$
\mathcal{L}_{KL} = H_e + \mathbb{E}_{(s,a)\sim\pi_e}\left[-\log \pi_a(a|s)\right]
$$

Lo stimatore Monte Carlo dal dataset $D=\{(s_i,a_i)\}_{i=1}^N$ è:

$$
\widehat{CE}(D) = \frac{1}{N}\sum_{i=1}^N -\log \pi_a(a_i \mid s_i)
$$

**Interpretazione:** $\widehat{CE}(D) = \mathcal{L}_{KL} - H_e$. Non conosciamo il valore assoluto della KL, ma il **decremento di $\widehat{CE}(D)$ nel tempo** è un indicatore diretto di convergenza (a meno di una costante additiva ignota e fissa).

### Implementazione

- **Action space discreto (softmax):** equivalente a `F.cross_entropy(logits, azioni)`. Nessun calcolo custom necessario.
- **Action space continuo, $\pi_a$ Gaussiana** con media $\mu_\theta(s)$ e varianza diagonale $\sigma_\theta^2(s)$:

$$
-\log \pi_a(a|s) = \frac{1}{2}\sum_{j=1}^d \left[\frac{(a_j - \mu_{\theta,j})^2}{\sigma_{\theta,j}^2} + \log \sigma_{\theta,j}^2\right] + \frac{d}{2}\log(2\pi)
$$

Implementato con `torch.distributions.Normal(mu, sigma).log_prob(a)`, poi si prende `-log_prob.mean()` sull'intero validation set.

### Nota importante

Non chiamarla `loss` nel logging, per evitare ambiguità con la loss di training (anche se matematicamente sono la stessa espressione). Nome usato: `val/cross_entropy_kl_proxy`.

## 2. RMSE su held-out

### Formula

$$
RMSE(D) = \sqrt{\frac{1}{N}\sum_{i=1}^N \| a_i - \mu_\theta(s_i) \|^2}
$$

dove $\mu_\theta(s_i)$ è l'azione **deterministica** predetta (media della Gaussiana, o output diretto se $\pi_a$ è deterministica) — mai un sample random, per garantire riproducibilità tra eval successive.

### Cosa aggiunge rispetto alla cross-entropy

La cross-entropy è in scala di log-probabilità, poco interpretabile in termini fisici. L'RMSE dà l'errore medio in unità reali dello spazio d'azione (es. gradi, Nm, ecc.), utile per capire *quanto* si sbaglia in pratica.

### Cosa NON cattura

L'RMSE ignora $\sigma_\theta$: una policy molto confidente ma leggermente disallineata e una policy molto incerta ma centrata correttamente danno lo stesso RMSE. La KL/cross-entropy penalizza la seconda situazione più duramente. Per questo le due metriche vanno lette insieme, non l'una al posto dell'altra.

Se le dimensioni dell'azione hanno scale molto diverse, viene loggato anche l'RMSE per-dimensione (`val/rmse_action_dim{j}`), oltre a quello aggregato (`val/rmse_action`).

## 3. Metrica di supporto: $\sigma$ media predetta

Se $\pi_a$ è Gaussiana, si logga anche `val/mean_predicted_std` = media di $\sigma_\theta(s)$ sul validation set.

**Perché serve:** se la cross-entropy scende ma l'RMSE resta stabile mentre $\sigma$ scende, la policy sta abbassando la cross-entropy solo aumentando la propria confidenza, senza un reale miglioramento nell'accuratezza della media predetta — un pattern di overfitting sulla varianza, individuabile solo incrociando le tre metriche.

## Riepilogo

| Metrica | Cosa misura | Limite |
|---|---|---|
| `val/cross_entropy_kl_proxy` | Convergenza probabilistica verso $\pi_e$ (proxy KL) | Valore assoluto non interpretabile ($H_e$ ignota); non distingue errore in media da errore in varianza |
| `val/rmse_action` (+ per-dim) | Errore fisico medio sull'azione | Ignora l'incertezza $\sigma$; fuorviante con azioni multimodali (collassa sulla media) |
| `val/mean_predicted_std` | Diagnostica overfitting sulla varianza | Solo per policy Gaussiane |

## Vincoli di implementazione

- Calcolo interamente sotto `torch.no_grad()`, nessuna backward call
- Media sull'intero validation set held-out, mai sui dati di training
- Funzione unica `compute_eval_metrics(policy, dataloader_val) -> dict`, chiamata solo nel loop di eval
