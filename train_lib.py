"""Shared loading / model definition."""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from app.features import build, CAT, NUM
D = "data/"
def load():
    prod, cust = pd.read_csv(D+"products.csv"), pd.read_csv(D+"customers.csv")
    tr = pd.read_csv(D+"train.csv", dtype={"delivery_pincode": str})
    tr = tr.sort_values("source").drop_duplicates("order_id", keep="first").sort_values("order_id").reset_index(drop=True)
    return build(tr, prod, cust), tr.returned.values, pd.to_datetime(tr.order_placed_at)
def logit():
    pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), CAT),
                             ("n", make_pipeline(SimpleImputer(), StandardScaler()), NUM)])
    return make_pipeline(pre, LogisticRegression(C=0.05, max_iter=3000))
