# Kestrel Home – returns risk

Scores every order for the chance it is returned, and tells the warehouse what to do about it.
**No API key, no paid service, no network calls.** Python 3.10+.

## Run it (clean machine)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# put the five CSVs from the data pack into ./data/  (train, test_unlabelled, sample_submission, products, customers)
python train.py            # ~10 s. Rebuilds model.joblib, predictions.csv, evidence/*.json|csv
python -m app.server       # open http://localhost:8000
```
`model.joblib` and `predictions.csv` are committed, so the server also starts without re-training
(it still needs `data/products.csv` and `data/customers.csv`). The data folder is git-ignored on purpose:
the ops policy (s10) forbids publishing customer data. **Keep this repository private.**

**Endpoint** – `POST /score`, one order as JSON (see the screen's example buttons):
```bash
curl -s localhost:8000/score -H 'Content-Type: application/json' -d '{"customer_id":"KC100000","sku":"KH-RV-02",
 "sales_channel":"marketplace","payment_mode":"cod","discount_pct":30,"qty":1,"order_value_inr":15399,
 "promised_delivery_days":9,"customer_prior_orders":3,"customer_prior_returns":2,"is_gift":"Y"}'
```
Returns probability, risk band, recommended action, plain-English reasons, data warnings. Bad input returns HTTP 400
with a readable message, never a stack trace.

## Layout
| path | what |
|---|---|
| `train.py`, `train_lib.py` | cleaning, validation, final fit, writes `predictions.csv` |
| `app/features.py` | the only place features are built (training and API share it) |
| `app/server.py`, `app/static/index.html` | endpoint + screen |
| `evidence/` | metrics, calibration, rupee analysis (`rupees.py` reproduces the memo numbers) |
| `EVIDENCE.md`, `DECISIONS.md`, `MEMO.md`, `AI_USE.md` | write-ups |

## Not used: AI model APIs. The product is a logistic regression; nothing in it can fail for lack of a key.
