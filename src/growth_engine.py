
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "model_artifacts"


def load_models(model_dir=DEFAULT_MODEL_DIR):
    """Load saved preprocessing and trained model artifacts."""
    model_dir = Path(model_dir)

    if not (model_dir / "preprocessor.joblib").exists():
        raise FileNotFoundError(f"No model artifacts in {model_dir}. Run `python src/train.py` first.")
    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    control_model = joblib.load(model_dir / "control_model.joblib")
    treatment_model = joblib.load(model_dir / "treatment_model.joblib")

    return preprocessor, control_model, treatment_model


def score_customers(customer_data, model_dir=DEFAULT_MODEL_DIR):
    """Score a customer batch and estimate incremental response."""
    if not isinstance(customer_data, pd.DataFrame):
        raise TypeError("customer_data must be a pandas DataFrame.")

    if "customer_id" not in customer_data.columns:
        raise ValueError("Missing required column: customer_id")

    if customer_data["customer_id"].isna().any():
        raise ValueError("customer_id cannot contain missing values.")

    if customer_data["customer_id"].duplicated().any():
        raise ValueError("customer_id must be unique within the input batch.")

    preprocessor, control_model, treatment_model = load_models(model_dir)
    required = list(preprocessor.feature_names_in_)

    missing = [col for col in required if col not in customer_data.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X = preprocessor.transform(customer_data[required])

    result = customer_data.copy()
    result["predicted_control_probability"] = (
        control_model.predict_proba(X)[:, 1]
    )
    result["predicted_treatment_probability"] = (
        treatment_model.predict_proba(X)[:, 1]
    )
    result["estimated_incremental_response"] = (
        result["predicted_treatment_probability"]
        - result["predicted_control_probability"]
    )

    return result


def score_csv(input_path, output_path, model_dir=DEFAULT_MODEL_DIR):
    """Read customer features from CSV, score them, and save results."""
    customer_data = pd.read_csv(input_path)
    scored = score_customers(customer_data, model_dir=model_dir)
    scored.to_csv(output_path, index=False)
    return scored


def offer_label(offer_amount, minimum_spend):
    """Human-readable action, e.g. '$5 off $30'."""
    return f"${offer_amount:g} off ${minimum_spend:g}"


def recommend_customers(
    scored_data,
    budget=100.0,
    offer_amount=5.0,
    minimum_spend=30.0,
    value_per_incremental_conversion=75.146327
):
    """Generate preliminary offer recommendations under a budget."""
    results = scored_data.copy()
    action = offer_label(offer_amount, minimum_spend)

    required = [
        "average_order_value",
        "predicted_treatment_probability",
        "estimated_incremental_response",
    ]
    missing = [col for col in required if col not in results.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if budget < 0:
        raise ValueError("budget cannot be negative.")

    results["estimated_basket_value"] = results["average_order_value"]
    results["offer_eligible"] = (
        results["estimated_basket_value"] >= minimum_spend
    )

    results["expected_incremental_conversions"] = (
        results["estimated_incremental_response"].clip(lower=0)
    )

    results["expected_incremental_revenue"] = (
        results["expected_incremental_conversions"]
        * value_per_incremental_conversion
    )

    results["expected_offer_cost"] = (
        results["predicted_treatment_probability"] * offer_amount
    )

    results["illustrative_net_value_after_offer_cost"] = (
        results["expected_incremental_revenue"]
        - results["expected_offer_cost"]
    )

    results["recommended_action"] = "No offer"
    results["recommendation_reason"] = (
        "Not selected under the current decision rules"
    )

    eligible = results[
        results["offer_eligible"]
        & (results["estimated_incremental_response"] > 0)
        & (results["illustrative_net_value_after_offer_cost"] > 0)
    ].sort_values(
        "illustrative_net_value_after_offer_cost",
        ascending=False
    )

    remaining_budget = float(budget)
    selected_indices = []

    for idx, row in eligible.iterrows():
        cost = float(row["expected_offer_cost"])
        if cost <= remaining_budget:
            selected_indices.append(idx)
            remaining_budget -= cost

    results.loc[selected_indices, "recommended_action"] = action
    results.loc[selected_indices, "recommendation_reason"] = (
        "Selected within budget using illustrative net value"
    )

    not_selected = eligible.index.difference(selected_indices)
    results.loc[not_selected, "recommendation_reason"] = (
        "Eligible but not selected within the available budget"
    )

    results["budget_used"] = results["expected_offer_cost"].where(
        results["recommended_action"] == action, 0.0
    )
    results["decision_budget"] = float(budget)

    return results
