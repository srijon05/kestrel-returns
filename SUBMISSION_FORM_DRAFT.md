# Submission form – draft answers (the form file was not in the pack)
- Expected score on hidden outcomes: AUC ~0.77 (0.74–0.80); accuracy ~89% at p≥0.5. See DECISIONS.md.
- What I used: Claude chat assistant; no paid API. Cost: Rs 0 incremental.
- What I discarded: gradient boosting, post-dispatch service/pickup columns (leakage), SKU/state/notes features.
- Assumptions: return cost Rs 1,150; hold cancel 12%; call prevents 35% (sensitivity 20%); margin on lost sale unknown (swept 0–30%).
- Known weaknesses: 95% accuracy not achievable; call-effect estimate comes from a non-randomised pilot; INSTALL_BOOKED unseen in training.
