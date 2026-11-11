import argparse
from pathlib import Path

import pandas as pd
from growth_engine import score_customers, recommend_customers, offer_label


def main():
    parser = argparse.ArgumentParser(
        description="Consumer Growth Decision Engine"
    )
    parser.add_argument("--input", required=True, help="Input customer CSV")
    parser.add_argument("--output", required=True, help="Output decision CSV")
    parser.add_argument("--budget", type=float, default=100.0, help="Expected discount spend allowed, in dollars")
    parser.add_argument("--offer-amount", type=float, default=5.0, help="Discount per redeemed coupon (default 5)")
    parser.add_argument("--min-spend", type=float, default=30.0, help="Minimum basket to qualify (default 30)")
    parser.add_argument("--value-per-conversion", type=float, default=75.146327,
                        help="Revenue (or margin) per incremental order; use your own figure")
    parser.add_argument("--model-dir", default=None, help="Folder with trained models (default: model_artifacts/)")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.is_file():
        parser.error(f"Input file not found: {input_path}")
    if args.budget < 0:
        parser.error("Budget cannot be negative.")
    if input_path == output_path:
        parser.error("Input and output paths must be different.")

    customer_data = pd.read_csv(input_path)
    kw = {"model_dir": args.model_dir} if args.model_dir else {}
    scored = score_customers(customer_data, **kw)
    decisions = recommend_customers(scored, budget=args.budget, offer_amount=args.offer_amount,
                                    minimum_spend=args.min_spend,
                                    value_per_incremental_conversion=args.value_per_conversion)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(output_path, index=False)

    selected = decisions[
        decisions["recommended_action"] == offer_label(args.offer_amount, args.min_spend)
    ]
    estimated_cost = selected["budget_used"].sum()

    print("\nConsumer Growth Decision Engine")
    print("--------------------------------")
    print(f"Customers evaluated: {len(decisions)}")
    print(f"Offer: {offer_label(args.offer_amount, args.min_spend)}")
    print(f"Customers recommended: {len(selected)}")
    print(f"Estimated offer cost: ${estimated_cost:.2f}")
    print(f"Budget remaining: ${args.budget - estimated_cost:.2f}")
    print(f"Decision file: {output_path}")


if __name__ == "__main__":
    main()
