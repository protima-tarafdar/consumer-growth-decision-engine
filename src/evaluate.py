"""Reproduce every number in the write-up from the data.

Usage:  python src/evaluate.py [--class-weight-balanced]
Writes outputs/results.json, outputs/*.csv and outputs/figures/*.png
"""
import argparse, json
import numpy as np, pandas as pd
from scipy import stats as st
from sklearn.metrics import roc_auc_score
from growth_model import load_experiment, split_arms, fit_two_models, score, ROOT

OUT = ROOT / "outputs"


def experiment_results(df):
    C, T = df[df.treatment_group == "Control"], df[df.treatment_group == "Treatment"]
    nc, nt = len(C), len(T); pc, pt = C.converted.mean(), T.converted.mean()
    d = pt - pc; se = np.sqrt(pc * (1 - pc) / nc + pt * (1 - pt) / nt)
    cv = C.post_intervention_value.sum() / nc
    inc_val = T.post_intervention_value.sum() - nt * cv
    cost = T.converted.sum() * 5.0
    res = dict(n_control=nc, n_treatment=nt, control_rate=pc, treatment_rate=pt, lift_pp=d * 100,
               ci95_pp=[(d - 1.96 * se) * 100, (d + 1.96 * se) * 100], z=d / se,
               incremental_conversions=d * nt, wasted_share=pc * nt / T.converted.sum(),
               incremental_value=inc_val, discount_cost=cost, net_value=inc_val - cost,
               roi_revenue_based=(inc_val - cost) / cost)
    rows = []
    for s in ["At Risk", "Lapsed", "Active", "New"]:
        c, t = C[C.segment == s], T[T.segment == s]
        iv = t.post_intervention_value.sum() - len(t) * c.post_intervention_value.sum() / len(c)
        cs = t.converted.sum() * 5.0
        rows.append(dict(segment=s, customers=len(c) + len(t), control_rate=c.converted.mean(),
                         treatment_rate=t.converted.mean(), lift_pp=(t.converted.mean() - c.converted.mean()) * 100,
                         wasted_share=c.converted.mean() / t.converted.mean(), net_value=iv - cs))
    return res, pd.DataFrame(rows)


def lift(x):
    return x[x.treatment_group == "Treatment"].converted.mean() - x[x.treatment_group == "Control"].converted.mean()


def heldout(df, balanced=False):
    parts = split_arms(df)
    ctrl_tr, ctrl_te = parts["Control"]; trt_tr, trt_te = parts["Treatment"]
    pre, ctrl, trt = fit_two_models(ctrl_tr, trt_tr, balanced)
    test = score(pd.concat([ctrl_te, trt_te], ignore_index=True), pre, ctrl, trt)
    auc = dict(control=roc_auc_score(ctrl_te.converted, score(ctrl_te, pre, ctrl, trt).predicted_control_probability),
               treatment=roc_auc_score(trt_te.converted, score(trt_te, pre, ctrl, trt).predicted_treatment_probability))
    test = test.sort_values("estimated_incremental_response", ascending=False).reset_index(drop=True)
    test["decile"] = pd.qcut(test["estimated_incremental_response"].rank(method="first", ascending=False), 10, labels=False) + 1
    dec = test.groupby("decile").apply(lambda g: pd.Series(dict(
        customers=len(g), predicted_lift_pp=g.estimated_incremental_response.mean() * 100, observed_lift_pp=lift(g) * 100)),
        include_groups=False).reset_index()
    def row(name, x, scale=1.0):
        c, t = x[x.treatment_group == "Control"], x[x.treatment_group == "Treatment"]
        iv = (t.post_intervention_value.mean() - c.post_intervention_value.mean()) * len(t); cs = t.converted.sum() * 5.0
        return dict(policy=name, customers=len(x), lift_pp=lift(x) * 100, incremental_conversions=lift(x) * len(t),
                    net_value=iv - cs, discount_cost=cs, return_per_dollar=(iv - cs) / cs)
    pol = [row(n, test[test.decile <= k]) for n, k in
           [("Top 10%", 1), ("Top 20%", 2), ("Top 30%", 3), ("Top 50%", 5), ("Top 70%", 7), ("Everyone", 10)]]
    # baselines: random targeting (expected value at the same reach) and a simple lifecycle rule
    ev = pol[-1]
    for frac in (0.3, 0.4):
        pol.append(dict(policy=f"Random {int(frac*100)}% (expected)", customers=int(len(test) * frac), lift_pp=ev["lift_pp"],
                        incremental_conversions=ev["incremental_conversions"] * frac, net_value=ev["net_value"] * frac,
                        discount_cost=ev["discount_cost"] * frac, return_per_dollar=ev["return_per_dollar"]))
    rule = test[test.segment.isin(["At Risk", "Lapsed"])]
    pol.append(row("Rule: At Risk + Lapsed", rule))
    # fair comparison: model-ranked customers at exactly the same reach as the rule
    ranked = test.sort_values("estimated_incremental_response", ascending=False)
    pol.append(row("Model, same reach as rule", ranked.head(len(rule))))
    rng = np.random.default_rng(7); top, rest = test[test.decile <= 3], test[test.decile > 3]
    bt = [lift(top.sample(len(top), replace=True, random_state=int(rng.integers(1e9)))) for _ in range(1000)]
    br = [lift(rest.sample(len(rest), replace=True, random_state=int(rng.integers(1e9)))) for _ in range(1000)]
    s = test.sort_values("estimated_incremental_response", ascending=False)
    isT = (s.treatment_group == "Treatment").astype(int).values; y = s.converted.values
    cT, cC = np.cumsum(isT), np.cumsum(1 - isT)
    q = np.cumsum(y * isT) - np.cumsum(y * (1 - isT)) * np.where(cC > 0, cT / np.maximum(cC, 1), 0)
    keep = ["customer_id", "segment", "acquisition_channel", "geography", "treatment_group", "converted", "post_intervention_value",
            "predicted_control_probability", "predicted_treatment_probability", "estimated_incremental_response", "decile"]
    test[keep].to_csv(OUT / "heldout_scores.csv", index=False)   # held-out customers only
    summary = dict(n_test=len(test), auc=auc, spearman_decile_vs_observed=float(st.spearmanr(dec.decile, dec.observed_lift_pp)[0]),
                   mean_predicted_control_prob=float(test.predicted_control_probability.mean()),
                   observed_control_rate=float(test[test.treatment_group == "Control"].converted.mean()),
                   top30_minus_rest_ci95_pp=list(np.percentile(np.array(bt) - np.array(br), [2.5, 97.5]) * 100),
                   top30_lift_pp=lift(top) * 100, top30_ci95_pp=list(np.percentile(bt, [2.5, 97.5]) * 100),
                   rest_lift_pp=lift(rest) * 100, rest_ci95_pp=list(np.percentile(br, [2.5, 97.5]) * 100),
                   top10_segment_mix=test[test.decile == 1].segment.value_counts(normalize=True).round(3).to_dict(),
                   top30_segment_mix=top.segment.value_counts(normalize=True).round(3).to_dict())
    return summary, dec, pd.DataFrame(pol), q


def figures(seg, dec, q):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    NAVY, CORAL, CTRL = "#14213D", "#E2725B", "#B4BDCD"
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    F = OUT / "figures"; F.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.3), dpi=200); xi = np.arange(len(seg)); w = .36
    ax.bar(xi - w / 2, seg.control_rate * 100, w, color=CTRL, label="Control"); ax.bar(xi + w / 2, seg.treatment_rate * 100, w, color=NAVY, label="Treatment")
    for i, r in seg.reset_index(drop=True).iterrows(): ax.text(i + w / 2, r.treatment_rate * 100 + .6, f"+{r.lift_pp:.1f} pp", ha="center", color=CORAL, fontweight="bold")
    ax.set_xticks(xi); ax.set_xticklabels(seg.segment); ax.set_ylabel("30-day conversion (%)"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(F / "conversion_by_segment.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.2, 3.3), dpi=200)
    ax.bar(dec.decile, dec.observed_lift_pp, .62, color=NAVY, label="Observed lift (held-out)")
    ax.plot(dec.decile, dec.predicted_lift_pp, color=CORAL, marker="o", label="Model-predicted response"); ax.axhline(0, color="#9AA3B2", lw=.8)
    ax.set_xticks(range(1, 11)); ax.set_xlabel("Decile by predicted response (1 = highest)"); ax.set_ylabel("Lift (pp)"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(F / "predicted_vs_observed_lift.png"); plt.close(fig)
    fr = np.arange(1, len(q) + 1) / len(q); fig, ax = plt.subplots(figsize=(5.2, 3.3), dpi=200)
    ax.plot(fr * 100, q, color=NAVY, label="Model-ranked targeting"); ax.plot([0, 100], [0, q[-1]], "--", color="#9AA3B2", label="Random targeting")
    ax.set_xlabel("Customers targeted, ranked by predicted response (%)"); ax.set_ylabel("Cumulative incremental conversions"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(F / "qini_curve.png"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--class-weight-balanced", action="store_true", help="reproduce the v1 (miscalibrated) setup"); a = ap.parse_args()
    OUT.mkdir(exist_ok=True); df = load_experiment()
    exp, seg = experiment_results(df); summ, dec, pol, q = heldout(df, balanced=a.class_weight_balanced)
    json.dump(dict(experiment=exp, heldout=summ), open(OUT / "results.json", "w"), indent=2, default=float)
    seg.to_csv(OUT / "segment_results.csv", index=False); dec.to_csv(OUT / "heldout_deciles.csv", index=False); pol.to_csv(OUT / "targeting_policies.csv", index=False)
    pd.DataFrame({"share_targeted": np.arange(1, len(q) + 1) / len(q), "cumulative_incremental_conversions": q}).to_csv(OUT / "qini_curve.csv", index=False)
    figures(seg, dec, q)
    print(json.dumps(dict(experiment=exp, heldout=summ), indent=2, default=float)); print(pol.round(2).to_string(index=False)); print(dec.round(2).to_string(index=False))
