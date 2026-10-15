# Splits

VERSION: `d986e975778a` (seed 0, corpus `2026.10.15-d7d3f451`)

Grouping: first 2 path levels within a document (D-024), refined deeper for groups > 1% of the pool. Held-out docs: eurlex/AMLD4. Post-cutoff: last_amended >= 2025-05-01 (D-025).

Fraction check (actual - target, of train/dev/test/silver pool): test +0.39 pp, dev +0.06 pp (within ±1 pp)

## Units per split x source x language

| split | source | language | units | postcutoff |
|---|---|---|---:|---:|
| train | ecfr | en | 19451 | 7595 |
| train | eurlex | en | 9685 | 24 |
| dev | ecfr | en | 1609 | 312 |
| dev | eurlex | en | 260 | 0 |
| test | ecfr | en | 2741 | 1131 |
| test | eurlex | en | 1100 | 0 |
| silver_val | ecfr | en | 1662 | 736 |
| silver_val | eurlex | en | 444 | 0 |
| heldout | eurlex | en | 641 | 0 |

## Totals

| split | units | postcutoff |
|---|---:|---:|
| train | 29136 | 7619 |
| dev | 1869 | 312 |
| test | 3841 | 1131 |
| silver_val | 2106 | 736 |
| heldout | 641 | 0 |
| gold | 0 | 0 |
| all | 37593 | 9798 |
