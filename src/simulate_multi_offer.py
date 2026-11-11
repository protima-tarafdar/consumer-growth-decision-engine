"""Simulate a randomized coupon test with several coupon amounts.

Usage:  python src/simulate_multi_offer.py
Writes data/multi_offer_experiment.csv (40,000 customers, seeded).

Customer features are resampled from data/consumer_growth_experiment.csv. Each customer is randomly
assigned to one of four arms: no coupon, $3 off, $5 off or $8 off a $30 order. All outcomes are simulated.

Ground truth (so results can be checked):
- Chance to order with no coupon depends on lifecycle segment and recent order volume.
- A coupon adds lift that grows with its size but levels off: lift = L_segment * h * (1 - exp(-amount / k_segment)).
  L = how much the segment can move at most; k = how big a coupon it takes to move them.
- h varies by customer inside a segment: customers whose order frequency is falling respond more.
  This gives a model something to learn beyond the segment label.
"""
import numpy as np
import pandas as pd
from growth_model import ROOT, FEATURES, load_experiment

ARMS = {"none": 0.0, "$3 off": 3.0, "$5 off": 5.0, "$8 off": 8.0}
MIN_SPEND = 30.0
BASE = {"Active": 0.10, "New": 0.03, "At Risk": 0.025, "Lapsed": 0.03}
MAX_LIFT = {"At Risk": 0.22, "Lapsed": 0.20, "New": 0.08, "Active": 0.05}
SENSITIVITY_K = {"At Risk": 3.0, "Lapsed": 9.0, "New": 5.0, "Active": 4.0}   # dollars


def true_probabilities(df):
    """Return a DataFrame of the true order probability under each arm."""
    z_orders = (df.orders_90d - df.orders_90d.mean()) / df.orders_90d.std()
    p0 = (df.segment.map(BASE) * np.exp(0.15 * z_orders)).clip(0.003, 0.4)
    z_trend = (df.frequency_trend - df.frequency_trend.mean()) / df.frequency_trend.std()
    h = (1 - 0.6 * z_trend).clip(0.3, 1.9)
    out = {}
    for arm, amount in ARMS.items():
        lift = df.segment.map(MAX_LIFT) * h * (1 - np.exp(-amount / df.segment.map(SENSITIVITY_K)))
        out[arm] = (p0 + lift).clip(0, 0.95)
    return pd.DataFrame(out, index=df.index)


def simulate(n=40_000, seed=7):
    rng = np.random.default_rng(seed)
    src = load_experiment()
    df = src[["customer_id"] + FEATURES].sample(n, replace=True, random_state=seed).reset_index(drop=True)
    df["customer_id"] = np.arange(1, n + 1)
    df["offer"] = rng.choice(list(ARMS), size=n)                       # equal 25% randomization
    df["offer_amount"] = df.offer.map(ARMS)
    probs = true_probabilities(df)
    p = probs.to_numpy()[np.arange(n), [list(ARMS).index(a) for a in df.offer]]
    df["ordered"] = (rng.random(n) < p).astype(int)
    basket = df.average_order_value * rng.lognormal(0, 0.25, n)
    basket = np.where(df.offer_amount > 0, np.maximum(basket, MIN_SPEND), basket)
    df["order_value"] = np.where(df.ordered == 1, basket.round(2), 0.0)
    for arm in ARMS:   # kept for checking only; never used as model input
        df[f"true_p_{arm.replace('$', '').replace(' off', '')}"] = probs[arm].round(5)
    return df


if __name__ == "__main__":
    out = simulate()
    path = ROOT / "data" / "multi_offer_experiment.csv"
    out.to_csv(path, index=False)
    print(f"Wrote {len(out):,} customers to {path}")
    print(out.groupby("offer").ordered.agg(["count", "mean"]).round(4))
