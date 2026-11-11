"""Shared modeling code: features, preprocessing, and the two-model uplift setup."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "consumer_growth_experiment.csv"

CATEGORICAL = ["acquisition_channel", "geography", "segment"]
NUMERICAL = [
    "customer_tenure_days", "orders_30d", "orders_previous_30d", "orders_90d",
    "total_orders", "total_spend", "average_order_value",
    "days_since_last_order", "frequency_trend",
]
FEATURES = ["customer_tenure_days", "acquisition_channel", "geography", "orders_30d",
            "orders_previous_30d", "orders_90d", "total_orders", "total_spend",
            "average_order_value", "days_since_last_order", "frequency_trend", "segment"]

RF_PARAMS = dict(n_estimators=300, max_depth=8, min_samples_leaf=20, random_state=42)


def load_experiment(path=DATA_PATH):
    """Load the experiment table (one row per customer)."""
    return pd.read_csv(path)


def build_preprocessor():
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("numerical", "passthrough", NUMERICAL),
    ])


def make_rf(balanced=False):
    """Unweighted by default. class_weight="balanced" (used in the first version of this
    project) inflates predicted probabilities (mean 0.37 vs 0.056 observed) and weakens
    the uplift ranking; see README, "What changed"."""
    return RandomForestClassifier(class_weight="balanced" if balanced else None, **RF_PARAMS)


def fit_two_models(control_df, treatment_df, balanced=False):
    """Fit preprocessing on the control arm, then one classifier per arm."""
    pre = build_preprocessor()
    Xc = pre.fit_transform(control_df[FEATURES])
    Xt = pre.transform(treatment_df[FEATURES])
    ctrl = make_rf(balanced).fit(Xc, control_df["converted"])
    trt = make_rf(balanced).fit(Xt, treatment_df["converted"])
    return pre, ctrl, trt


def score(df, pre, ctrl, trt):
    X = pre.transform(df[FEATURES])
    out = df.copy()
    out["predicted_control_probability"] = ctrl.predict_proba(X)[:, 1]
    out["predicted_treatment_probability"] = trt.predict_proba(X)[:, 1]
    out["estimated_incremental_response"] = (
        out["predicted_treatment_probability"] - out["predicted_control_probability"])
    return out


def split_arms(df, test_size=0.25, seed=42):
    """Stratified 75/25 split within each experiment arm."""
    parts = {}
    for arm in ["Control", "Treatment"]:
        d = df[df["treatment_group"] == arm]
        parts[arm] = train_test_split(d, test_size=test_size, random_state=seed, stratify=d["converted"])
    return parts
