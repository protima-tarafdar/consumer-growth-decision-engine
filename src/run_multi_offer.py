"""Recommend a coupon amount (or none) for each customer, within a budget.

Usage:
    python src/multi_offer.py                       # trains the per-amount models (once)
    python src/run_multi_offer.py --input data/sample_customer_input.csv \
        --output outputs/multi_offer_decisions.csv --budget 100 --value-per-order 12
"""
import argparse
import joblib
import pandas as pd
from growth_model import ROOT, FEATURES
from multi_offer import ART, predict_arm_probs, choose_offers


def main():
    ap = argparse.ArgumentParser(description="Choose a coupon amount per customer")
    ap.add_argument("--input", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--budget", type=float, default=None, help="Expected coupon spend allowed, in dollars (default: no cap)")
    ap.add_argument("--value-per-order", type=float, default=12.0, help="What the platform earns per order (fees + commission)")
    a = ap.parse_args()
    path = ART / "arm_models.joblib"
    if not path.exists():
        raise SystemExit("No models found. Run `python src/multi_offer.py` first.")
    bundle = joblib.load(path)
    df = pd.read_csv(a.input)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")
    if a.budget is not None and a.budget < 0:
        raise SystemExit("Budget cannot be negative.")
    probs = predict_arm_probs(df, bundle["preprocessor"], bundle["models"])
    dec = choose_offers(probs, a.value_per_order, a.budget)
    out = pd.concat([df[["customer_id", "segment"]], probs.add_prefix("p_order_").round(4), dec.round(4)], axis=1)
    out.to_csv(a.output, index=False)
    print(out.recommended_offer.value_counts().to_string())
    print(f"Expected coupon cost: ${out.expected_coupon_cost.sum():.2f}   Expected net value: ${out.expected_net_value.sum():.2f}")


if __name__ == "__main__":
    main()
