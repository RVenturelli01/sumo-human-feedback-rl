# Run mancanti per analisi più robuste

## Configurazioni esistenti

| famiglia    | queries | temperature | seed |
|-------------|---------|-------------|------|
| bernoulli   | 10k     | 20          | 0,1,2 |
| bernoulli   | 50k     | 20          | 0,1,2 |
| bernoulli   | 100k    | 20          | 0,1,2 |
| binary      | 10k     | 20          | 0,1,2 |
| binary      | 50k     | 20          | 0,1,2 |
| soft        | 2k      | 20          | 0,1,2 |
| soft        | 5k      | 20          | 0,1,2 |
| soft        | 10k     | **5**       | 0,1,2 |
| soft        | 10k     | 20          | 0,1,2 |
| soft        | 10k     | **50**      | 0,1,2 |

---

## Priorità 1 — Completare il sweep di temperatura (12 run)

**Problema:** il grafico "Effetto della temperatura" mostra la curva completa (T=5/20/50) solo per q=10k.
Per 2k e 5k query c'è un solo punto (T=20), quindi non si può concludere che T=20 sia ottimale indipendentemente dal budget.

| famiglia | queries | temperature | seed | motivo |
|----------|---------|-------------|------|--------|
| soft | 2k | 5  | 0,1,2 | sweep temperatura a budget basso |
| soft | 2k | 50 | 0,1,2 | sweep temperatura a budget basso |
| soft | 5k | 5  | 0,1,2 | sweep temperatura a budget medio |
| soft | 5k | 50 | 0,1,2 | sweep temperatura a budget medio |

**Cosa si guadagna:** poter affermare "T=20 è il punto ottimale per tutti i budget testati" con dati diretti.

---

## Priorità 2 — Confronto equo tra famiglie a stesso budget (12 run)

**Problema:** binary e bernoulli sono stati testati solo a q=10k e q≥50k.
Per confrontare soft vs binary vs bernoulli a *parità di query* servono configurazioni allineate.

| famiglia  | queries | temperature | seed | motivo |
|-----------|---------|-------------|------|--------|
| binary    | 2k      | 20          | 0,1,2 | confronto diretto con soft q=2k |
| binary    | 5k      | 20          | 0,1,2 | confronto diretto con soft q=5k |
| bernoulli | 2k      | 20          | 0,1,2 | confronto diretto con soft q=2k |
| bernoulli | 5k      | 20          | 0,1,2 | confronto diretto con soft q=5k |

**Cosa si guadagna:** scatter e tabella riassuntiva con righe allineate per budget — argomento più diretto per la tesi ("a parità di query, soft supera le altre").

---

## Priorità 3 — Soft a budget più alti (6 run)

**Problema:** soft è stato testato fino a q=10k, mentre bernoulli arriva a 100k.
Non si sa se soft continua a migliorare o satura.

| famiglia | queries | temperature | seed | motivo |
|----------|---------|-------------|------|--------|
| soft | 50k  | 20 | 0,1,2 | confronto con bernoulli 50k e binary 50k |
| soft | 100k | 20 | 0,1,2 | confronto con bernoulli 100k, vede se satura |

**Cosa si guadagna:** curva di scaling completa per soft; verifica se a budget elevati bernoulli recupera terreno.

---

## Priorità 4 — Temperatura intermedia per soft q=10k (3 run)

**Problema:** con solo T=5/20/50 la curva ha tre punti — l'ottimo potrebbe cadere tra 5 e 20 o tra 20 e 50.

| famiglia | queries | temperature | seed | motivo |
|----------|---------|-------------|------|--------|
| soft | 10k | 10 | 0,1,2 | raffina il minimo tra T=5 e T=20 |

**Cosa si guadagna:** curva temperatura più liscia; utile solo se il paper include una sezione dedicata all'ablazione della temperatura.

---

## Riepilogo

| priorità | n. run aggiuntive | impatto |
|----------|-------------------|---------|
| 1 — sweep temperatura completo | 12 | **alto** — necessario per claim sulla temperatura |
| 2 — confronto equo a stesso budget | 12 | **alto** — necessario per confronto principale |
| 3 — soft a budget più alti | 6  | medio — utile per curva di scaling |
| 4 — temperatura intermedia | 3  | basso — solo per ablazione dettagliata |
| **totale minimo (P1+P2)** | **24** | |
| **totale completo** | **33** | |
