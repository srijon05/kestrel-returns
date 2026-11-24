"""Kestrel returns-risk service.  Run:  python -m app.server   ->  http://localhost:8000
No external API / key is used anywhere. POST /score takes ONE order as JSON."""
import os, sys, joblib, numpy as np, pandas as pd
from flask import Flask, jsonify, request, send_from_directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from app.features import build, FEATURES, CAT, NUM
from train_lib import load

PROD = pd.read_csv(os.path.join(ROOT, "data/products.csv"))
CUST = pd.read_csv(os.path.join(ROOT, "data/customers.csv"))
MODEL = joblib.load(os.path.join(ROOT, "model.joblib"))["model"]
PRE, CLF = MODEL.steps[0][1], MODEL.steps[1][1]
_X, _, _ = load()
_Z = PRE.transform(_X[FEATURES]); _MEAN = np.asarray(_Z.mean(axis=0)).ravel()
_NAMES = list(PRE.get_feature_names_out())
CALL_THRESHOLD = 0.112   # break-even: Rs45 call vs 35% x Rs1,150 prevented (policy s4, s7)
HIGH = 0.30

TEXT = {  # (feature) -> sentence builder
    "customer_prior_returns": lambda r: f"Customer has returned {int(r.customer_prior_returns)} earlier order(s)",
    "prior_return_rate": lambda r: f"Past return rate is {r.prior_return_rate:.0%} of their orders",
    "shield": lambda r: "Kestrel Shield member (free returns make returning easy)",
    "is_cod": lambda r: "Cash on delivery (COD orders come back about twice as often)",
    "promised_delivery_days": lambda r: f"Long delivery promise ({int(r.promised_delivery_days)} days)",
    "discount_pct": lambda r: f"Deep discount ({int(r.discount_pct)}%)",
    "list_price": lambda r: f"High-ticket product ({r.family}, list Rs {int(r.list_price):,})",
    "is_gift": lambda r: "Order marked as a gift",
    "sales_channel": lambda r: f"Sold via {r.sales_channel}",
    "payment_mode": lambda r: f"Payment mode: {r.payment_mode}",
    "family": lambda r: f"Product family: {r.family}",
    "customer_prior_orders": lambda r: f"Customer has {int(r.customer_prior_orders)} prior order(s)",
}

NEGF = {"promised_delivery_days": lambda r: f"Short delivery promise ({int(r.promised_delivery_days)} days)",
        "discount_pct": lambda r: f"Small discount ({int(r.discount_pct)}%)",
        "list_price": lambda r: f"Lower-ticket product (list Rs {int(r.list_price):,})"}
SKIP = {"month", "hour", "dow", "tenure_days", "net_value", "qty", "slow_promise", "has_prior_return", "no_address"}
NEG = {"shield": "Not a Shield member", "is_cod": "Paid in advance (prepaid orders return less)",
       "is_gift": "Not a gift", "customer_prior_returns": "Customer has no / few earlier returns"}

def explain(row: pd.Series, k=4):
    z = PRE.transform(row.to_frame().T[FEATURES])
    contrib = np.asarray(z.todense() if hasattr(z, "todense") else z).ravel()
    contrib = (contrib - _MEAN) * CLF.coef_.ravel()
    agg = {}
    for name, c in zip(_NAMES, contrib):
        f = name.split("__", 1)[1]
        f = next((x for x in CAT if f.startswith(x + "_")), f)
        f = "customer_prior_returns" if f == "prior_return_rate" else f   # one "return history" reason
        agg[f] = agg.get(f, 0) + c
    items = sorted(agg.items(), key=lambda kv: -abs(kv[1]))
    items = [(f, c) for f, c in items if f not in SKIP]
    def txt(f, c):
        if c < 0 and f in NEGF: return NEGF[f](row)
        if c < 0 and f in NEG:
            return NEG[f] if f != "customer_prior_returns" or row.customer_prior_returns == 0 else None
        return TEXT.get(f, lambda r, f=f: f)(row)
    up = [{"reason": txt(f, c), "effect": "raises risk", "weight": round(float(c), 2)} for f, c in items if c > 0.05][:k]
    down = [{"reason": txt(f, c), "effect": "lowers risk", "weight": round(float(c), 2)} for f, c in items if c < -0.05]
    down = [d for d in down if d["reason"]][:2]
    return up, down

def score_record(rec: dict):
    need = ["customer_id", "sku", "sales_channel", "payment_mode", "discount_pct", "qty", "order_value_inr",
            "promised_delivery_days", "customer_prior_orders", "customer_prior_returns"]
    miss = [k for k in need if rec.get(k) in (None, "")]
    if miss: raise ValueError("missing field(s): " + ", ".join(miss))
    if rec["sku"] not in set(PROD.sku): raise ValueError(f"unknown sku {rec['sku']}")
    if rec["customer_id"] not in set(CUST.customer_id): raise ValueError(f"unknown customer_id {rec['customer_id']}")
    d = {"order_id": rec.get("order_id", "NEW"), "order_placed_at": rec.get("order_placed_at") or pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
         "delivery_pincode": str(rec.get("delivery_pincode") or "000000"), "is_gift": rec.get("is_gift", "N"),
         "delivery_note": rec.get("delivery_note", "")}
    for k in need: d[k] = rec[k]
    for k in ("discount_pct", "qty", "order_value_inr", "promised_delivery_days", "customer_prior_orders", "customer_prior_returns"):
        d[k] = float(d[k])
    row = build(pd.DataFrame([d]), PROD, CUST).iloc[0]
    p = float(MODEL.predict_proba(row.to_frame().T[FEATURES])[0, 1])
    up, down = explain(row)
    if p >= HIGH: action, why = "CALL_BEFORE_DISPATCH", "High risk. Confirm model, address and intent by phone before dispatch. Do not hold: holds cost more than they save."
    elif p >= CALL_THRESHOLD: action, why = "CALL_BEFORE_DISPATCH", "Above break-even for a Rs 45 confirmation call."
    else: action, why = "SHIP", "Below break-even: a call costs more than it is expected to save."
    warn = []
    if row.no_address: warn.append("No address captured (walk-in order, default pincode) - delivery details cannot be checked.")
    if row.order_value_inr > 2 * row.order_value_fixed or row.order_value_fixed != row.order_value_inr:
        warn.append("Order value looked ~100x list price; treated as paise (payment-gateway issue) and corrected.")
    return {"order_id": d["order_id"], "return_probability": round(p, 4), "score": round(p, 4),
            "risk_band": "high" if p >= HIGH else "medium" if p >= CALL_THRESHOLD else "low",
            "recommended_action": action, "action_explanation": why, "reasons": up, "reassuring_factors": down,
            "data_warnings": warn, "model": "logistic regression, trained 1 Apr 2025 - 30 Jun 2026", "expected_cost_of_return_inr": round(p * 1150)}

app = Flask(__name__, static_folder=os.path.join(ROOT, "app", "static"))

@app.get("/")
def index(): return send_from_directory(app.static_folder, "index.html")

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/options")
def options():
    return {"skus": PROD[["sku", "model_name"]].to_dict("records"), "sample_customers": CUST.head(5).customer_id.tolist()}

@app.post("/score")
def score():
    try:
        rec = request.get_json(force=True, silent=False)
        if not isinstance(rec, dict): raise ValueError("body must be a single JSON object")
        return jsonify(score_record(rec))
    except Exception as e:  # polite failure, never a stack trace
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
