"""Feature engineering shared by training, batch scoring and the API.

Only information that exists at dispatch time is used. Deliberately EXCLUDED:
last_service_event_type, pickup_scheduled_at (written after delivery / after the return is
approved - the test file only ever contains NONE / INSTALL_BOOKED, a state the training file
never shows), and `source` (an import artefact).
"""
import numpy as np
import pandas as pd

CAT = ["family", "sales_channel", "payment_mode"]  # sku/state/note_type tested: pure noise (see evidence)
NUM = ["discount_pct", "qty", "promised_delivery_days", "customer_prior_orders",
       "customer_prior_returns", "prior_return_rate", "has_prior_return", "no_address",
       "is_gift", "shield", "list_price", "net_value", "tenure_days", "hour", "dow",
       "month", "is_cod", "slow_promise"]
FEATURES = CAT + NUM


def fix_values(df: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """The October-2025 gateway stored paise (x100). Rather than patch one month, compare the
    stored value with list price x qty x (1-discount) and divide by 100 where it is ~100x."""
    df = df.merge(products[["sku", "family", "list_price_inr"]], on="sku", how="left")
    expected = df.list_price_inr * df.qty * (1 - df.discount_pct / 100)
    ratio = df.order_value_inr / expected
    df["order_value_fixed"] = np.where(ratio > 50, df.order_value_inr / 100, df.order_value_inr)
    return df


def note_type(s: pd.Series) -> pd.Series:
    s = s.fillna("").str.lower()
    out = pd.Series("none", index=s.index)
    for key, pat in [("neighbour", "neighbour"), ("call", "call"), ("fragile", "fragile"),
                     ("security", "security"), ("office", "office"), ("floor", "floor"),
                     ("whatsapp", "whatsapp"), ("morning", "morning"), ("after", "deliver after"),
                     ("ring", "ring"), ("landmark", "landmark"), ("gate", "gate")]:
        out[s.str.contains(pat)] = key
    return out


def build(orders: pd.DataFrame, products: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    df = fix_values(orders.copy(), products)
    df = df.merge(customers, on="customer_id", how="left")
    ts = pd.to_datetime(df.order_placed_at)
    df["hour"], df["dow"], df["month"] = ts.dt.hour, ts.dt.dayofweek, ts.dt.month
    df["tenure_days"] = (ts - pd.to_datetime(df.signup_date)).dt.days.fillna(-1)
    df["shield"] = (df.shield_member == "Y").astype(int)
    df["is_gift"] = (df.is_gift == "Y").astype(int)
    df["is_cod"] = (df.payment_mode == "cod").astype(int)
    df["no_address"] = (df.delivery_pincode.astype(str).str.zfill(6) == "000000").astype(int)
    df["prior_return_rate"] = df.customer_prior_returns / df.customer_prior_orders.replace(0, np.nan)
    df["has_prior_return"] = (df.customer_prior_returns > 0).astype(int)
    df["list_price"] = df.list_price_inr
    df["net_value"] = df.order_value_fixed
    df["slow_promise"] = (df.promised_delivery_days >= 8).astype(int)
    df["note_type"] = note_type(df.delivery_note)
    for c in CAT:
        df[c] = df[c].fillna("unk").astype(str)
    df["note_type"] = df["note_type"].astype(str)
    return df
