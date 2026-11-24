# Evidence

All numbers reproduce with `python train.py` and `python evidence/rupees.py`. Validation is **time-ordered**: model trained on
orders before 1 Apr 2026, scored on Apr–Jun 2026 (2,126 orders, 11.5% returned) – the same "train on past, score the future"
situation as the July–September file. Random k-fold would have flattered the model.

## How good is it?
| | Apr–Jun 2026 hold-out |
|---|---|
| Return rate | 11.5% |
| Accuracy of "flag nothing" | **88.5%** (free; this is the bar any model must beat) |
| Best accuracy any threshold of my model reaches | **89.8%** |
| AUC (ranking quality, 0.5 = coin flip) | **0.786** (rolling windows: 0.768 / 0.783 / 0.786) |
| Precision-recall AUC (baseline 0.115) | 0.416 |
| Calibration | top-fifth predicted 31.5% vs actual 30.8%; bottom fifth 2.2% vs 2.3% |
| Top 10% of orders by score | 43% are returned (3.7x base rate); catches 37% of all returns |
| Top 2% | 76% returned |

**It does not reach 95% accuracy and no honest model on this data will.** Returns are 1-in-9 orders and most are not predictable
from what is known at checkout; ceiling on this data is ~90% accuracy. A model reaches "98.4%" only by cheating (below).

## How often is it wrong?
At the cut-off where it calls an order a return (p ≥ 0.5): it is right 66% of the time when it says "return", and it finds
13% of the returns. At the cost-optimal call cut-off (p ≥ 11%, ~32% of orders flagged) about 1 in 4 flagged orders is a real return;
3 in 4 are false alarms – acceptable only because a call costs Rs 45 and a missed return costs Rs 1,150.

## Traps in the pack, and what I did
| Finding | Evidence | Action |
|---|---|---|
| `pickup_scheduled_at`, `REVERSE_PICKUP`, `TECH_VISIT`, `INSTALL_DONE`, `DEMO_DONE` are **after-the-fact** | pickup present ⇒ 92% returned, REVERSE_PICKUP ⇒ 100%; policy s7 says they're written *after* return approval; export is "as of export day". Using the pickup column alone gives **98.4% accuracy** in train | excluded all service/pickup columns |
| Test file contains only `NONE` / `INSTALL_BOOKED`; `INSTALL_BOOKED` never occurs in train | value counts | would be an unseen category in production even if it were legitimate |
| Order values for **Oct 2025 are ×100** (new gateway) | stored/(list×qty×(1−disc)) = 100.0 for every Oct row, 1.0 elsewhere | corrected by ratio test (also done inside the API) |
| **651 duplicate order_ids** from partner feed | all have a CRM twin; every other column identical | kept CRM row |
| Walk-in orders with pincode 000000 (889 train rows) | README | flag only; location is not used |
| Legacy-Zoho resolution events stored in UTC | policy s9 | irrelevant: those columns are excluded |
| Customer `Shield` has **2x the return rate** (18.7% vs 9.3%) | train | kept as a feature, see memo |
| Email says Rs 600 per return; policy and Finance say **Rs 1,150** | | used 1,150 |

## What I tried and dropped
- Gradient-boosted trees: 0.756 AUC vs logistic 0.786 on the same split – overfit; signal is additive. Kept as comparison in `train.py`.
- SKU, state, parsed delivery-note, hour/day: no lift (permutation importance ≈ 0); removed from the model's categorical set.
- Strongest real signals: customer's prior returns, Shield, promised delivery days, COD, price, discount depth.
