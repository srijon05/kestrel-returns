"""Train + validate. Usage: python train.py   (writes model.joblib, predictions.csv, evidence/metrics.json)"""
import json, joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from app.features import build, CAT, NUM, FEATURES

D = "data/"
prod, cust = pd.read_csv(D + "products.csv"), pd.read_csv(D + "customers.csv")
tr = pd.read_csv(D + "train.csv", dtype={"delivery_pincode": str})
te = pd.read_csv(D + "test_unlabelled.csv", dtype={"delivery_pincode": str})

# 1. de-duplicate: partner-feed re-imports are copies of CRM rows (all 651 have a CRM twin)
tr = tr.sort_values("source").drop_duplicates("order_id", keep="first").sort_values("order_id").reset_index(drop=True)
Xtr_all, Xte = build(tr, prod, cust), build(te, prod, cust)
y = tr.returned.values
t = pd.to_datetime(tr.order_placed_at)

def gbm():
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=250, max_depth=4, min_samples_leaf=40,
                                          l2_regularization=2.0, categorical_features=[FEATURES.index(c) for c in CAT],
                                          random_state=0)
def prep(df):
    X = df[FEATURES].copy()
    for c in CAT: X[c] = X[c].astype("category")
    return X
# categorical codes must be consistent between fit and predict -> fix categories from train
cats = {c: sorted(Xtr_all[c].unique()) for c in CAT}
def enc(df):
    X = df[FEATURES].copy()
    for c in CAT: X[c] = pd.Categorical(X[c], categories=cats[c]).codes
    return X
ident = lambda d: d[FEATURES]
def logit():
    pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), CAT),
                             ("n", make_pipeline(SimpleImputer(), StandardScaler()), NUM)])
    return make_pipeline(pre, LogisticRegression(C=0.05, max_iter=3000))

def cost_table(p, yv, hold_margin=0.20, value=None):
    """Policy: act on top-k% scored orders. Returns rupee effect per 1,000 orders."""
    rows = []
    order = np.argsort(-p)
    n = len(p)
    for k in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
        idx = order[: int(k * n)]
        rets = yv[idx].sum()
        # CALL: Rs45 each, prevents 35% of returns among called orders (each prevented return saves Rs1,150)
        call = 1150 * 0.35 * rets - 45 * len(idx)
        # HOLD: 12% cancel. Cancel loses margin on the order (assumed 20% of value, see memo); a held order that
        # was going to return and cancels saves the Rs1,150 return; a held non-returner that cancels loses margin.
        v = value[idx] if value is not None else np.full(len(idx), 5000.0)
        cancel = 0.12
        hold = sum(cancel * (1150 * yv[i] - hold_margin * v[j] * (1 - yv[i])) for j, i in enumerate(idx))
        # (holding does not itself stop the 88% who stay; a held returner who stays still returns => no saving)
        rows.append(dict(top_pct=int(k * 100), orders=len(idx), returns_caught=int(rets),
                         precision=round(rets / len(idx), 3), call_net_inr=round(call), hold_net_inr=round(hold)))
    return pd.DataFrame(rows)

# 2. Time-ordered validation: train through 2026-03, validate Apr-Jun 2026 (mimics "train on past, score the future")
cut = pd.Timestamp("2026-04-01")
a, b = (t < cut).values, (t >= cut).values
res = {}
base = y[b].mean()
res["val_rows"], res["val_base_rate"] = int(b.sum()), round(float(base), 4)
res["accuracy_always_no_return"] = round(1 - base, 4)
for name, mk, f in [("logistic", logit, lambda d: d[FEATURES]), ("gbm", gbm, enc)]:
    m = mk().fit(f(Xtr_all[a]), y[a]); p = m.predict_proba(f(Xtr_all[b]))[:, 1]
    best = max(((p >= th) == y[b]).mean() for th in np.linspace(0.05, 0.95, 91))
    res[name] = dict(auc=round(roc_auc_score(y[b], p), 4), pr_auc=round(average_precision_score(y[b], p), 4),
                     brier=round(brier_score_loss(y[b], p), 4), best_possible_accuracy=round(float(best), 4),
                     acc_at_0_5=round(float(((p >= .5) == y[b]).mean()), 4),
                     recall_at_0_5=round(float(((p >= .5) & (y[b] == 1)).sum() / y[b].sum()), 4),
                     precision_at_0_5=round(float(((p >= .5) & (y[b] == 1)).sum() / max(1, (p >= .5).sum())), 4))
    if name == "logistic":
        pv = p
# 3. Rolling-origin check for stability
roll = []
for c0, c1 in [("2025-12-01", "2026-02-01"), ("2026-02-01", "2026-04-01"), ("2026-04-01", "2026-07-01")]:
    aa, bb = (t < pd.Timestamp(c0)).values, ((t >= pd.Timestamp(c0)) & (t < pd.Timestamp(c1))).values
    m = logit().fit(Xtr_all[aa][FEATURES], y[aa]); p = m.predict_proba(Xtr_all[bb][FEATURES])[:, 1]
    roll.append(dict(window=f"{c0}..{c1}", auc=round(roc_auc_score(y[bb], p), 4)))
res["rolling_auc"] = roll
# 4. What the leak would have given (to show Ritu why "98% accuracy" is meaningless)
leak = tr.pickup_scheduled_at.notna().astype(int)
res["LEAK_demo_accuracy_using_pickup_column"] = round(float((leak == y).mean()), 4)
# 5. Calibration + rupees on validation
Xv = Xtr_all[b].reset_index(drop=True)
ct = cost_table(pv, y[b], value=Xv.net_value.values)
ct.to_csv("evidence/cost_by_top_pct_validation.csv", index=False)
q = pd.qcut(pv, 5, duplicates="drop"); cal = pd.DataFrame({"p": pv, "y": y[b]}).groupby(q, observed=True).agg(mean_pred=("p", "mean"), actual=("y", "mean"), n=("y", "size")).round(3)
cal.to_csv("evidence/calibration_validation.csv")
# Shield segment check
sh = Xv.shield.values == 1
res["auc_shield"] = round(roc_auc_score(y[b][sh], pv[sh]), 4); res["auc_nonshield"] = round(roc_auc_score(y[b][~sh], pv[~sh]), 4)
res["shield_share_of_top10pct"] = round(float(sh[np.argsort(-pv)[: int(.1 * len(pv))]].mean()), 3)
res["shield_share_of_validation"] = round(float(sh.mean()), 3)

# 6. Final fit on all history, score test
final = logit().fit(Xtr_all[FEATURES], y)
Xte["score"] = final.predict_proba(Xte[FEATURES])[:, 1]
joblib.dump(dict(model=final, features=FEATURES, base_rate=float(y.mean())), "model.joblib")
sub = Xte[["order_id", "score"]].round(5)
assert len(sub) == 2096 and sub.order_id.is_unique
sub.to_csv("predictions.csv", index=False)
res["test_score_mean"], res["test_rows"] = round(float(sub.score.mean()), 4), len(sub)
json.dump(res, open("evidence/metrics.json", "w"), indent=2)
print(json.dumps(res, indent=2)); print(ct.to_string()); print(cal.to_string())
