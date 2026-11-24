# AI tools – honest account
- **Used:** Claude (Anthropic) as an analysis/coding assistant in this chat session for data exploration, the modelling script, the Flask service and these write-ups.
- **Cost:** no API spend; part of an existing chat subscription. The shipped product calls no model API; running cost is Rs 0.
- **Discarded:** gradient boosting (worse out-of-time), the service/pickup features (leakage), SKU/state/note features (noise), a first reason-text version that mislabelled "not Shield" as "Shield".
- **Verification I did:** every number in the memo is produced by `train.py` / `evidence/rupees.py`; API smoke-tested on good, risky, and malformed input.
