# Splits

VERSION: `5b7c2d0af82e` (seed 0, corpus `2026.10.15-4ce7830f`)

Grouping: first 2 path levels within a document (D-024), refined deeper for groups > 1% of the pool. Held-out docs: eurlex/AMLD4. Post-cutoff: last_amended >= 2025-05-01 (D-025).

Fraction check (actual - target, of train/dev/test/silver pool): test +0.22 pp, dev -0.00 pp (within ±1 pp)

## Units per split x source x language

| split | source | language | units | postcutoff |
|---|---|---|---:|---:|
| train | ecfr | en | 20090 | 7786 |
| train | eurlex | en | 9413 | 24 |
| train | thai | en | 388 | 0 |
| train | thai | th | 4651 | 0 |
| dev | ecfr | en | 1620 | 332 |
| dev | eurlex | en | 227 | 0 |
| dev | thai | en | 38 | 0 |
| dev | thai | th | 270 | 0 |
| test | ecfr | en | 2725 | 1125 |
| test | eurlex | en | 1053 | 0 |
| test | thai | en | 48 | 0 |
| test | thai | th | 581 | 0 |
| silver_val | ecfr | en | 855 | 475 |
| silver_val | eurlex | en | 729 | 0 |
| silver_val | thai | en | 43 | 0 |
| silver_val | thai | th | 373 | 0 |
| heldout | eurlex | en | 641 | 0 |
| gold | ecfr | en | 173 | 56 |
| gold | eurlex | en | 67 | 0 |
| gold | thai | th | 60 | 0 |

## Totals

| split | units | postcutoff |
|---|---:|---:|
| train | 34542 | 7810 |
| dev | 2155 | 332 |
| test | 4407 | 1125 |
| silver_val | 2000 | 475 |
| heldout | 641 | 0 |
| gold | 300 | 56 |
| all | 44045 | 9798 |
