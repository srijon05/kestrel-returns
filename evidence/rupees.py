"""Rupee analysis on the Apr-Jun 2026 hold-out (model trained on data before 1 Apr 2026). Run: python evidence/rupees.py"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, json
from sklearn.metrics import roc_auc_score
from train_lib import load
X, y, t = load()
a, b = (t < "2026-04-01").values, (t >= "2026-04-01").values
from train_lib import logit
from app.features import FEATURES
p = logit().fit(X[a][FEATURES], y[a]).predict_proba(X[b][FEATURES])[:, 1]; yv = y[b]; n = len(yv)
RET, CALL = 1150, 45
out = {}
for prevent in (0.20, 0.35):
    be = CALL / (prevent * RET)                      # call an order if P(return) > this
    sel = p >= be
    net = prevent * RET * yv[sel].sum() - CALL * sel.sum()
    out[f"call_prevent_{int(prevent*100)}pct"] = dict(break_even_prob=round(be, 3), orders_called=int(sel.sum()), pct_called=round(sel.mean()*100, 1),
        returns_in_called=int(yv[sel].sum()), net_inr_per_quarter=round(float(net)), call_all_net=round(float(prevent*RET*yv.sum() - CALL*n)))
# bootstrap CI for the 35% case
rng = np.random.default_rng(0); be = CALL/(0.35*RET); nets = []
for _ in range(1000):
    i = rng.integers(0, n, n); s = p[i] >= be
    nets.append(0.35*RET*yv[i][s].sum() - CALL*s.sum())
out["call_35_net_95ci"] = [round(float(np.percentile(nets, 2.5))), round(float(np.percentile(nets, 97.5))) ]
# HOLD: every assumption spelled out. Margin on lost sale is NOT in the pack -> sweep it.
sel = p >= 0.30; v = X[b].net_value.values
for m in (0.0, 0.10, 0.20, 0.30):
    ret_saved = 0.12 * RET * yv[sel].sum()          # a held returner who cancels = return avoided
    lost = m * 0.12 * (v[sel] * (1 - yv[sel])).sum() # a held good order that cancels = margin lost
    out[f"hold_p>=0.30_margin_{int(m*100)}pct"] = round(float(ret_saved - lost))
out["hold_p>=0.30_orders"] = int(sel.sum()); out["hold_precision"] = round(float(yv[sel].mean()), 3)
# Accuracy bar
for th in (0.5, 0.3, 0.112):
    out[f"accuracy_if_flagging_p>={th}"] = round(float(((p >= th) == yv).mean()), 4)
out["accuracy_flag_nothing"] = round(float(1 - yv.mean()), 4)
# Shield: who gets called
sh = X[b].shield.values == 1; be = CALL/(0.35*RET)
out["shield_called_share"] = round(float((sh & (p >= be)).sum()/ (p >= be).sum()), 3)
json.dump(out, open("evidence/rupees.json", "w"), indent=2); print(json.dumps(out, indent=2))
