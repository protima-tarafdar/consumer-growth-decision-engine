"""Choose the best coupon amount for each customer, then compare policies.

Usage:  python src/multi_offer.py [--value-per-order 12]
Reads data/multi_offer_experiment.csv, trains one model per arm on 70% of customers, and evaluates
policies on the held-out 30%. Writes outputs/multi_offer_*.csv and a figure.

Decision rule: for each customer and each coupon amount a,
    expected net value(a) = (P(order | a) - P(order | none)) * V  -  P(order | a) * a
V is what the platform earns per order (fees + commission). The coupon cost a is paid on every
redeemed order, including orders that would have happened anyway. The customer gets the amount with
the highest expected net value, or no coupon if none is positive.
"""
import argparse, json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from growth_model import ROOT, FEATURES, build_preprocessor, make_rf

DATA = ROOT / "data" / "multi_offer_experiment.csv"
OUT = ROOT / "outputs"
ART = ROOT / "model_artifacts" / "multi_offer"
ARMS = ["none", "$3 off", "$5 off", "$8 off"]
AMOUNT = {"none": 0.0, "$3 off": 3.0, "$5 off": 5.0, "$8 off": 8.0}
TRUE_COL = {"none": "true_p_none", "$3 off": "true_p_3", "$5 off": "true_p_5", "$8 off": "true_p_8"}


def fit_arm_models(train):
    pre = build_preprocessor().fit(train[FEATURES])
    models = {}
    for arm in ARMS:
        d = train[train.offer == arm]
        models[arm] = make_rf().fit(pre.transform(d[FEATURES]), d.ordered)
    return pre, models


def predict_arm_probs(df, pre, models):
    X = pre.transform(df[FEATURES])
    return pd.DataFrame({arm: m.predict_proba(X)[:, 1] for arm, m in models.items()}, index=df.index)


def net_values(probs, value_per_order):
    """Expected incremental net value of each coupon vs no coupon, per customer."""
    return pd.DataFrame({arm: (probs[arm] - probs["none"]) * value_per_order - probs[arm] * AMOUNT[arm]
                         for arm in ARMS if arm != "none"}, index=probs.index)


def choose_offers(probs, value_per_order, budget=None):
    """Best coupon per customer (or none). With a budget, keep the most valuable until expected cost is spent."""
    nv = net_values(probs, value_per_order)
    best = nv.idxmax(axis=1); best_val = nv.max(axis=1)
    choice = best.where(best_val > 0, "none")
    cost = pd.Series([probs.at[i, a] * AMOUNT[a] if a != "none" else 0.0 for i, a in choice.items()], index=choice.index)
    if budget is not None:
        order = best_val.where(choice != "none", -np.inf).sort_values(ascending=False).index
        spent = 0.0; keep = set()
        for i in order:
            if choice[i] == "none": break
            if spent + cost[i] <= budget: keep.add(i); spent += cost[i]
        choice = choice.where(choice.index.isin(keep), "none"); cost = cost.where(choice != "none", 0.0)
    return pd.DataFrame({"recommended_offer": choice, "expected_net_value": best_val.where(choice != "none", 0.0),
                         "expected_coupon_cost": cost})


def true_policy_value(df, policy, value_per_order):
    """Exact expected result of a policy, using the simulator's true probabilities (possible only because data is simulated)."""
    p_none = df[TRUE_COL["none"]]
    p = pd.Series([df.at[i, TRUE_COL[a]] for i, a in policy.items()], index=policy.index)
    amt = policy.map(AMOUNT)
    return dict(incremental_orders=float((p - p_none).sum()),
                coupon_cost=float((p * amt).sum()),
                net_value=float(((p - p_none) * value_per_order - p * amt).sum()),
                coupons_sent=int((policy != "none").sum()))


def ipw_policy_value(df, policy, value_per_order):
    """Estimate from observed outcomes only: customers whose random arm matches the policy, reweighted (25% per arm)."""
    match = (df.offer == policy).astype(float)
    net = df.ordered * (value_per_order - df.offer_amount)
    base = df[df.offer == "none"].ordered.mean() * value_per_order
    return float((match * net / 0.25).mean() - base) * len(df)


def segment_rule(train, value_per_order):
    """Baseline: one coupon amount per segment, picked from training-set averages."""
    rule = {}
    for seg, g in train.groupby("segment"):
        r = g.groupby("offer").ordered.mean()
        nv = {a: (r[a] - r["none"]) * value_per_order - r[a] * AMOUNT[a] for a in ARMS if a != "none"}
        best = max(nv, key=nv.get); rule[seg] = best if nv[best] > 0 else "none"
    return rule


def run(value_per_order=12.0, seed=42):
    df = pd.read_csv(DATA)
    train, test = train_test_split(df, test_size=0.3, random_state=seed, stratify=df.offer)
    pre, models = fit_arm_models(train)
    ART.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(preprocessor=pre, models=models), ART / "arm_models.joblib")
    probs = predict_arm_probs(test, pre, models)
    rule = segment_rule(train, value_per_order)
    true_nv = net_values(test[[TRUE_COL[a] for a in ARMS]].set_axis(ARMS, axis=1), value_per_order)
    policies = {
        "No coupons": pd.Series("none", index=test.index),
        "Everyone $3": pd.Series("$3 off", index=test.index),
        "Everyone $5": pd.Series("$5 off", index=test.index),
        "Everyone $8": pd.Series("$8 off", index=test.index),
        "Best amount per segment": test.segment.map(rule),
        "Personalized (model)": choose_offers(probs, value_per_order).recommended_offer,
        "Perfect knowledge (ceiling)": true_nv.idxmax(axis=1).where(true_nv.max(axis=1) > 0, "none"),
    }
    rows = []
    for name, pol in policies.items():
        t = true_policy_value(test, pol, value_per_order)
        rows.append(dict(policy=name, **t, ipw_net_value_estimate=ipw_policy_value(test, pol, value_per_order) if name != "No coupons" else 0.0))
    res = pd.DataFrame(rows)
    res["net_value_per_customer"] = res.net_value / len(test)
    pers = policies["Personalized (model)"]
    mix = pd.crosstab(test.segment, pers, normalize="index").reindex(columns=ARMS, fill_value=0).round(3)
    agree = float((pers == policies["Perfect knowledge (ceiling)"]).mean())
    summary = dict(n_test=len(test), value_per_order=value_per_order, segment_rule=rule, agreement_with_ceiling=agree,
                   offer_mix=pers.value_counts(normalize=True).round(3).to_dict())
    return res, mix, summary, test.assign(**{f"pred_{a}": probs[a] for a in ARMS}, recommended_offer=pers)


def figure(res, mix):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    GREEN, CORAL, SAGE = "#1F3A1C", "#E2725B", "#C7CFBC"
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    r = res[res.policy != "No coupons"]
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=200)
    cols = [CORAL if p == "Personalized (model)" else (SAGE if "ceiling" in p else GREEN) for p in r.policy]
    ax.barh(r.policy[::-1], r.net_value[::-1], color=cols[::-1]); ax.axvline(0, color="#999", lw=.8)
    ax.set_xlabel("Net value vs. no coupons, held-out customers ($)")
    fig.tight_layout(); (OUT / "figures").mkdir(parents=True, exist_ok=True); fig.savefig(OUT / "figures" / "multi_offer_policies.png"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--value-per-order", type=float, default=12.0); a = ap.parse_args()
    res, mix, summary, scored = run(a.value_per_order)
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "multi_offer_policies.csv", index=False); mix.to_csv(OUT / "multi_offer_mix_by_segment.csv")
    json.dump(summary, open(OUT / "multi_offer_summary.json", "w"), indent=2)
    figure(res, mix)
    pd.set_option("display.width", 200)
    print(res.round(1).to_string(index=False)); print(mix); print(json.dumps(summary, indent=1))
