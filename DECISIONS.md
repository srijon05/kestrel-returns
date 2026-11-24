# Decisions I made where the brief was silent

1. **"95% accuracy" is not the target.** Flagging nothing scores 88.5%; the honest ceiling is ~90%. I optimised ranking (AUC) and rupees, and tell Ritu so in the memo.
2. **Return cost = Rs 1,150** (policy s4, Finance) not Rs 600 (Ritu's estimate).
3. **No service/pickup columns.** They are written after dispatch. Using them is leakage that would give 98% in testing and nothing in production.
4. **"Hold" vs "call".** Hold: a held order cancels 12% of the time (policy s7). Margin lost on a cancelled sale is not in the pack; I swept 0–30%. Hold only pays at ≤~7% margin. A Rs 45 call that prevents 35% of returns (spring pilot) is positive down to p = 11%. I recommend calls, not holds.
5. **Pilot's 35% is soft** ("seemed to work", one season, not randomised). I show 20% as the downside case and recommend a randomised test.
6. **Shield:** not excluded (they genuinely return more) but not held either; calling is non-punitive. Ritu said "deal with Shield later" – I flag that 41% of recommended calls would land on Shield members, so the script matters.
7. **Time-ordered validation** and de-duplication before any splitting (duplicates would leak across a random split).
8. **Per-order cost:** model runs locally; Rs 0 per order (Farhan's condition).
9. **Submission form:** not included in the pack I received (the brief says "complete [form]" but no file is attached). Draft answers are in `SUBMISSION_FORM_DRAFT.md`; please transfer them.

## Expected score (written before submitting)
Unseen July–Sep 2026 orders: **AUC 0.77 (range 0.74–0.80)**, PR-AUC ≈ 0.40. Accuracy ≈ 89% at a 0.5 threshold, i.e. within ~1 point of "predict no returns". Reasons: three rolling windows gave 0.77–0.79; test is one quarter further out, so I shade down; the labels I cannot see could also differ if behaviour drifts (e.g. the test period has more `INSTALL_BOOKED` orders, which I can't model).
